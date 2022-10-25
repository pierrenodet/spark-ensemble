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
import org.apache.spark.ml.boosting.BoostingParams
import org.apache.spark.ml.ensemble.EnsemblePredictionModelType
import org.apache.spark.ml.ensemble.EnsembleRegressorType
import org.apache.spark.ml.ensemble.HasBaseLearner
import org.apache.spark.ml.ensemble.Utils
import org.apache.spark.ml.linalg.BLAS
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.rdd.util.PeriodicRDDCheckpointer
import org.apache.spark.sql.Dataset
import org.apache.spark.storage.StorageLevel
import org.json4s.DefaultFormats
import org.json4s.JObject
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.JsonMethods.render

import java.util.Locale

private[ml] trait BoostingRegressorParams extends BoostingParams[EnsembleRegressorType] {

  /**
   * Loss function which Boosting tries to minimize. (case-insensitive) Supported: "exponential",
   * "linear", "squared". (default = exponential)
   *
   * @group param
   */
  val lossType: Param[String] =
    new Param(
      this,
      "lossType",
      "loss function, exponential by default",
      (value: String) =>
        BoostingRegressorParams.supportedLossTypes.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getLossType: String = $(lossType).toLowerCase(Locale.ROOT)

  setDefault(lossType -> "exponential")

  /**
   * Voting strategy to aggregate predictions of base regressor. (case-insensitive) Supported:
   * "median", "mean". (default = median)
   *
   * @group param
   */
  val votingStrategy: Param[String] =
    new Param(
      this,
      "votingStrategy",
      "voting strategy, (case-insensitive). Supported options:" + s"${BoostingRegressorParams.supportedVotingStrategy
          .mkString(",")}",
      (value: String) =>
        BoostingRegressorParams.supportedVotingStrategy.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getVotingStrategy: String = $(votingStrategy).toLowerCase(Locale.ROOT)

  setDefault(votingStrategy -> "median")

}

private[ml] object BoostingRegressorParams {

  final val supportedLossTypes: Array[String] =
    Array("exponential", "squared", "linear").map(_.toLowerCase(Locale.ROOT))

  final val supportedVotingStrategy: Array[String] =
    Array("median", "mean").map(_.toLowerCase(Locale.ROOT))

  def loss(lossType: String): Double => Double = lossType match {
    case "exponential" =>
      error => 1 - math.exp(-error)
    case "linear" =>
      error => error
    case "squared" =>
      error => math.pow(error, 2)
    case _ => throw new RuntimeException(s"Boosting was given bad loss type: $lossType")

  }

  def saveImpl(
      instance: BoostingRegressorParams,
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

class BoostingRegressor(override val uid: String)
    extends Regressor[Vector, BoostingRegressor, BoostingRegressionModel]
    with BoostingRegressorParams
    with MLWritable {

  def setBaseLearner(value: EnsembleRegressorType): this.type =
    set(baseLearner, value)

  /** @group setParam */
  def setNumBaseLearners(value: Int): this.type = set(numBaseLearners, value)

  /** @group setParam */
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /** @group setParam */
  def setLossType(value: String): this.type = set(lossType, value)

  /** @group setParam */
  def setVotingStrategy(value: String): this.type = set(votingStrategy, value)

  def this() = this(Identifiable.randomUID("BoostingRegressor"))

  override def copy(extra: ParamMap): BoostingRegressor = {
    val copied = new BoostingRegressor(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  def error(label: Double, prediction: Double): Double = math.abs(label - prediction)

  def loss(error: Double): Double = BoostingRegressorParams.loss($(lossType))(error)

  override protected def train(dataset: Dataset[_]): BoostingRegressionModel = instrumented {
    instr =>
      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(
        this,
        labelCol,
        featuresCol,
        predictionCol,
        weightCol,
        lossType,
        numBaseLearners,
        checkpointInterval)

      val spark = dataset.sparkSession
      val sc = spark.sparkContext

      val instances = extractInstances(dataset)

      instances
        .persist(StorageLevel.MEMORY_AND_DISK)
      instances.count()

      val featuresMetadata = Utils.getFeaturesMetadata(dataset, $(featuresCol))

      val models = Array.ofDim[EnsemblePredictionModelType]($(numBaseLearners))
      val estimatorWeights = Array.ofDim[Double]($(numBaseLearners))

      var boostingWeights = instances.map(_.weight)
      val boostingWeightsCheckpointer = new PeriodicRDDCheckpointer[Double](
        $(checkpointInterval),
        sc,
        StorageLevel.MEMORY_AND_DISK)
      boostingWeightsCheckpointer.update(boostingWeights)

      var sumWeights = boostingWeights.treeReduce((_ + _), $(aggregationDepth))

      var i = 0
      var best = 0
      var done = false

      while (i < $(numBaseLearners) && !done && (sumWeights > 0)) {

        instr.logNamedValue("iteration", i)

        val weighted =
          instances.zip(boostingWeights).map { case (instance, boostingWeight) =>
            instance.copy(weight = boostingWeight / sumWeights)
          }

        val df = spark
          .createDataFrame(weighted)
          .withMetadata("features", featuresMetadata)

        val model =
          fitBaseLearner($(baseLearner), "label", "features", $(predictionCol), Some("weight"))(
            df)

        val errors =
          instances.map(instance => error(instance.label, model.predict(instance.features)))

        val maxError = errors.treeReduce((_ max _), $(aggregationDepth))

        val losses = if (maxError == 0) {
          best = i
          done = true
          errors.map(loss)
        } else {
          errors.map(error => loss(error / maxError))
        }

        val estimatorError = weighted
          .zip(losses)
          .treeAggregate(0d)(
            { case (acc, (instance, loss)) => acc + instance.weight * loss },
            { _ + _ },
            $(aggregationDepth))

        if (estimatorError >= 0.5) { best = i - 1; done = true }

        val beta = estimatorError / (1 - estimatorError)
        val estimatorWeight = if (beta == 0.0) 1.0 else math.log(1.0 / beta)

        boostingWeights = weighted
          .zip(losses)
          .map { case (instance, loss) =>
            instance.weight * math.pow(beta, 1 - loss)
          }
        boostingWeightsCheckpointer.update(boostingWeights)

        sumWeights = boostingWeights.treeReduce((_ + _), $(aggregationDepth))

        estimatorWeights(i) = estimatorWeight
        models(i) = model

        best = i
        i += 1

      }

      best += 1

      boostingWeightsCheckpointer.unpersistDataSet()
      boostingWeightsCheckpointer.deleteAllCheckpoints()

      instances.unpersist()

      new BoostingRegressionModel(estimatorWeights.take(best), models.take(best))

  }

  override def write: MLWriter = new BoostingRegressor.BoostingRegressorWriter(this)

}

object BoostingRegressor extends MLReadable[BoostingRegressor] {

  override def read: MLReader[BoostingRegressor] = new BoostingRegressorReader

  override def load(path: String): BoostingRegressor = super.load(path)

  private[BoostingRegressor] class BoostingRegressorWriter(instance: BoostingRegressor)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      BoostingRegressorParams.saveImpl(instance, path, sc)
    }

  }

  private class BoostingRegressorReader extends MLReader[BoostingRegressor] {

    /** Checked against metadata when loading model */
    private val className = classOf[BoostingRegressor].getName

    override def load(path: String): BoostingRegressor = {
      val (metadata, learner) = BoostingRegressorParams.loadImpl(path, sc, className)
      val bc = new BoostingRegressor(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

class BoostingRegressionModel(
    override val uid: String,
    val weights: Array[Double],
    val models: Array[EnsemblePredictionModelType])
    extends RegressionModel[Vector, BoostingRegressionModel]
    with BoostingRegressorParams
    with MLWritable {

  def this(weights: Array[Double], models: Array[EnsemblePredictionModelType]) =
    this(Identifiable.randomUID("BoostingRegressionModel"), weights, models)

  val numModels = models.size

  private val sumWeights: Double = weights.sum

  private def predictWeightedMedian(features: Vector): Double = {
    val predictions = models.map(_.predict(features))
    Utils.weightedMedian(predictions, weights)
  }

  private def predictWeightedMean(features: Vector): Double = {
    BLAS.dot(Vectors.dense(models.map(_.predict(features))), Vectors.dense(weights)) / sumWeights
  }

  override def predict(features: Vector): Double = {
    $(votingStrategy) match {
      case "median" => predictWeightedMedian(features)
      case "mean" => predictWeightedMean(features)
    }
  }

  override def copy(extra: ParamMap): BoostingRegressionModel = {
    val copied = new BoostingRegressionModel(uid, weights, models)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter =
    new BoostingRegressionModel.BoostingRegressionModelWriter(this)

}

object BoostingRegressionModel extends MLReadable[BoostingRegressionModel] {

  override def read: MLReader[BoostingRegressionModel] = new BoostingRegressionModelReader

  override def load(path: String): BoostingRegressionModel = super.load(path)

  private[BoostingRegressionModel] class BoostingRegressionModelWriter(
      instance: BoostingRegressionModel)
      extends MLWriter {

    private case class Data(weight: Double)

    override protected def saveImpl(path: String): Unit = {
      BoostingRegressorParams.saveImpl(
        instance,
        path,
        sc,
        Some("numModels" -> instance.numModels))
      instance.models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach { case (model, idx) =>
        val modelPath = new Path(path, s"model-$idx").toString
        model.save(modelPath)
      }
      instance.weights.zipWithIndex.foreach { case (weight, idx) =>
        val data = Data(weight)
        val dataPath = new Path(path, s"data-$idx").toString
        sparkSession.createDataFrame(Seq(data)).repartition(1).write.json(dataPath)
      }

    }
  }

  private class BoostingRegressionModelReader extends MLReader[BoostingRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BoostingRegressionModel].getName

    override def load(path: String): BoostingRegressionModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = BoostingRegressorParams.loadImpl(path, sc, className)
      val numModels = (metadata.metadata \ "numModels").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
      }
      val boostsData = (0 until numModels).toArray.map { idx =>
        val dataPath = new Path(path, s"data-$idx").toString
        val data = sparkSession.read.json(dataPath).select("weight").head()
        data.getAs[Double](0)
      }
      val bcModel =
        new BoostingRegressionModel(metadata.uid, boostsData, models)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
