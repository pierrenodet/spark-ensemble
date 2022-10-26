/*
 * Copyright 2019 Pierre Nodet
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.regression

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.bagging.BaggingParams
import org.apache.spark.ml.ensemble.EnsemblePredictionModelType
import org.apache.spark.ml.ensemble.EnsembleRegressorType
import org.apache.spark.ml.ensemble.HasBaseLearner
import org.apache.spark.ml.ensemble.Utils
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.ThreadUtils
import org.json4s.DefaultFormats
import org.json4s.JObject
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.concurrent.Future
import scala.concurrent.duration.Duration

private[ml] trait BaggingRegressorParams extends BaggingParams[EnsembleRegressorType] {}

private[ml] object BaggingRegressorParams {

  def saveImpl(
      instance: BaggingRegressorParams,
      path: String,
      sc: SparkContext,
      extraMetadata: Option[JObject] = None): Unit = {

    val params = instance.extractParamMap().toSeq
    val jsonParams = render(
      params
        .filter { case ParamPair(p, _) => p.name != "baseLearner" }
        .map { case ParamPair(p, v) => p.name -> parse(p.jsonEncode(v)) }
        .toList)

    DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, Some(jsonParams))
    HasBaseLearner.saveImpl(instance, path, sc)

  }

  def loadImpl(
      path: String,
      sc: SparkContext,
      expectedClassName: String): (DefaultParamsReader.Metadata, EnsembleRegressorType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val learner = HasBaseLearner.loadImpl[EnsembleRegressorType](path, sc)
    (metadata, learner)

  }

}

class BaggingRegressor(override val uid: String)
    extends Regressor[Vector, BaggingRegressor, BaggingRegressionModel]
    with BaggingRegressorParams
    with MLWritable {

  def this() = this(Identifiable.randomUID("BaggingRegressor"))

  /** @group setParam */
  def setBaseLearner(value: EnsembleRegressorType): this.type =
    set(baseLearner, value)

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /** @group setParam */
  def setReplacement(value: Boolean): this.type = set(replacement, value)

  /** @group setParam */
  def setSubsampleRatio(value: Double): this.type = set(subsampleRatio, value)

  /** @group setParam */
  def setSubspaceRatio(value: Double): this.type = set(subspaceRatio, value)

  /** @group setParam */
  def setNumBaseLearners(value: Int): this.type = set(numBaseLearners, value)

  /**
   * Set the maximum level of parallelism to evaluate models in parallel. Default is 1 for serial
   * evaluation
   *
   * @group expertSetParam
   */
  def setParallelism(value: Int): this.type = set(parallelism, value)

  override def copy(extra: ParamMap): BaggingRegressor = {
    val copied = new BaggingRegressor(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  override protected def train(dataset: Dataset[_]): BaggingRegressionModel = instrumented {
    instr =>
      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(
        this,
        labelCol,
        weightCol,
        featuresCol,
        predictionCol,
        numBaseLearners,
        subsampleRatio,
        replacement,
        subspaceRatio,
        seed)

      val spark = dataset.sparkSession
      val sc = spark.sparkContext

      val instances = extractInstances(dataset)
      instances.persist(StorageLevel.MEMORY_AND_DISK)

      val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))

      val subspaces =
        Array.tabulate($(numBaseLearners))(i =>
          subspace($(subspaceRatio), numFeatures, $(seed) + i))

      val futureModels = subspaces
        .map(subspace =>
          Future[EnsemblePredictionModelType] {

            val bagged = instances
              .sample($(replacement), $(subsampleRatio), $(seed))
            val subbagged =
              bagged.map(instance => instance.copy(features = slice(subspace)(instance.features)))

            val featuresMetadata =
              Utils.getFeaturesMetadata(dataset, $(featuresCol), Some(subspace))

            val df = spark
              .createDataFrame(subbagged)
              .withMetadata("features", featuresMetadata)

            fitBaseLearner($(baseLearner), "label", "features", $(predictionCol), Some("weight"))(
              df)

          }(getExecutionContext))

      val models = futureModels.map(ThreadUtils.awaitResult(_, Duration.Inf))

      instances.unpersist()

      new BaggingRegressionModel(subspaces, models)

  }

  override def write: MLWriter = new BaggingRegressor.BaggingRegressorWriter(this)

}

object BaggingRegressor extends MLReadable[BaggingRegressor] {

  override def read: MLReader[BaggingRegressor] = new BaggingRegressorReader

  override def load(path: String): BaggingRegressor = super.load(path)

  private[BaggingRegressor] class BaggingRegressorWriter(instance: BaggingRegressor)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      BaggingRegressorParams.saveImpl(instance, path, sc)
    }

  }

  private class BaggingRegressorReader extends MLReader[BaggingRegressor] {

    /** Checked against metadata when loading model */
    private val className = classOf[BaggingRegressor].getName

    override def load(path: String): BaggingRegressor = {
      val (metadata, learner) = BaggingRegressorParams.loadImpl(path, sc, className)
      val bc = new BaggingRegressor(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

class BaggingRegressionModel(
    override val uid: String,
    val subspaces: Array[Array[Int]],
    val models: Array[EnsemblePredictionModelType])
    extends RegressionModel[Vector, BaggingRegressionModel]
    with BaggingRegressorParams
    with MLWritable {

  def this(subspaces: Array[Array[Int]], models: Array[EnsemblePredictionModelType]) =
    this(Identifiable.randomUID("BaggingRegressionModel"), subspaces, models)

  val numModels: Int = models.length

  override def predict(features: Vector): Double = {
    var sum = 0d
    var i = 0
    while (i < numModels) {
      sum += models(i).predict(slice(subspaces(i))(features)); i += 1
    }
    sum / numModels
  }

  override def copy(extra: ParamMap): BaggingRegressionModel = {
    val copied = new BaggingRegressionModel(uid, subspaces, models)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new BaggingRegressionModel.BaggingRegressionModelWriter(this)

}

object BaggingRegressionModel extends MLReadable[BaggingRegressionModel] {

  override def read: MLReader[BaggingRegressionModel] = new BaggingRegressionModelReader

  override def load(path: String): BaggingRegressionModel = super.load(path)

  private[BaggingRegressionModel] class BaggingRegressionModelWriter(
      instance: BaggingRegressionModel)
      extends MLWriter {

    private case class Data(subspace: Array[Int])

    override protected def saveImpl(path: String): Unit = {
      val extraJson = "numModels" -> instance.numModels
      BaggingRegressorParams.saveImpl(instance, path, sc, Some(extraJson))
      instance.models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach { case (model, idx) =>
        val modelPath = new Path(path, s"model-$idx").toString
        model.save(modelPath)
      }
      instance.subspaces.zipWithIndex.foreach { case (subspace, idx) =>
        val data = Data(subspace)
        val dataPath = new Path(path, s"data-$idx").toString
        sparkSession.createDataFrame(Seq(data)).repartition(1).write.json(dataPath)
      }

    }
  }

  private class BaggingRegressionModelReader extends MLReader[BaggingRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BaggingRegressionModel].getName

    override def load(path: String): BaggingRegressionModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, baseLearner) = BaggingRegressorParams.loadImpl(path, sc, className)
      val numModels = metadata.getParamValue("numBaseLearners").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
      }
      val subspaces = (0 until numModels).toArray.map { idx =>
        val dataPath = new Path(path, s"data-$idx").toString
        val data = sparkSession.read.json(dataPath).select("subspace").head()
        data.getAs[Seq[Long]](0).map(_.toInt).toArray
      }
      val brModel = new BaggingRegressionModel(metadata.uid, subspaces, models)
      metadata.getAndSetParams(brModel)
      brModel.set("baseLearner", baseLearner)
      brModel
    }
  }
}
