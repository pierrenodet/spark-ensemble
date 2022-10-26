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

package org.apache.spark.ml.classification

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.SparkException
import org.apache.spark.ml.bagging._
import org.apache.spark.ml.ensemble._
import org.apache.spark.ml.linalg.BLAS
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql.Dataset
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.ThreadUtils
import org.json4s.DefaultFormats
import org.json4s.JObject
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import java.util.Locale
import scala.concurrent.Future
import scala.concurrent.duration.Duration

private[ml] trait BaggingClassifierParams
    extends BaggingParams[EnsembleClassifierType]
    with ClassifierParams {

  /**
   * Voting strategy to aggregate predictions of base classifiers. (case-insensitive) Supported:
   * "hard", "soft". (default = hard)
   *
   * @group param
   */
  val votingStrategy: Param[String] =
    new Param(
      this,
      "votingStrategy",
      "voting strategy, (case-insensitive). Supported options:" + s"${BaggingClassifierParams.supportedVotingStrategy
          .mkString(",")}",
      (value: String) =>
        BaggingClassifierParams.supportedVotingStrategy.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getVotingStrategy: String = $(votingStrategy).toLowerCase(Locale.ROOT)

  setDefault(votingStrategy -> "hard")
}

private[ml] object BaggingClassifierParams {

  final val supportedVotingStrategy: Array[String] =
    Array("soft", "hard").map(_.toLowerCase(Locale.ROOT))

  def saveImpl(
      instance: BaggingClassifierParams,
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
      expectedClassName: String): (DefaultParamsReader.Metadata, EnsembleClassifierType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val learner = HasBaseLearner.loadImpl[EnsembleClassifierType](path, sc)
    (metadata, learner)

  }

}

class BaggingClassifier(override val uid: String)
    extends ProbabilisticClassifier[Vector, BaggingClassifier, BaggingClassificationModel]
    with BaggingClassifierParams
    with MLWritable {

  def this() = this(Identifiable.randomUID("BaggingClassifier"))

  /** @group setParam */
  def setBaseLearner(value: EnsembleClassifierType): this.type =
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
  def setVotingStrategy(value: String): this.type = set(votingStrategy, value)

  /** @group setParam */
  def setNumBaseLearners(value: Int): this.type = set(numBaseLearners, value)

  /**
   * Set the maximum level of parallelism to evaluate models in parallel. Default is 1 for serial
   * evaluation
   *
   * @group expertSetParam
   */
  def setParallelism(value: Int): this.type = set(parallelism, value)

  override def copy(extra: ParamMap): BaggingClassifier = {
    val copied = new BaggingClassifier(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  override protected def train(dataset: Dataset[_]): BaggingClassificationModel = instrumented {
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

      val numFeatures = MetadataUtils.getNumFeatures(dataset, getFeaturesCol)
      val numClasses = getNumClasses(dataset)
      instr.logNumClasses(numClasses)
      validateNumClasses(numClasses)

      val subspaces =
        Array.tabulate($(numBaseLearners))(i =>
          subspace($(subspaceRatio), numFeatures, $(seed) + i))

      val futureModels = subspaces
        .map(subspace =>
          Future[EnsemblePredictionModelType] {

            val bag = instances
              .sample($(replacement), $(subsampleRatio), $(seed))
            val subbag =
              bag.map(instance => instance.copy(features = slice(subspace)(instance.features)))

            val featuresMetadata =
              Utils.getFeaturesMetadata(dataset, $(featuresCol), Some(subspace))

            val df = spark
              .createDataFrame(subbag)
              .withMetadata("features", featuresMetadata)

            fitBaseLearner($(baseLearner), "label", "features", $(predictionCol), Some("weight"))(
              df)

          }(getExecutionContext))

      val models = futureModels.map(ThreadUtils.awaitResult(_, Duration.Inf))

      instances.unpersist()

      new BaggingClassificationModel(numClasses, subspaces, models)

  }

  override def write: MLWriter = new BaggingClassifier.BaggingClassifierWriter(this)

}

object BaggingClassifier extends MLReadable[BaggingClassifier] {

  override def read: MLReader[BaggingClassifier] = new BaggingClassifierReader

  override def load(path: String): BaggingClassifier = super.load(path)

  private[BaggingClassifier] class BaggingClassifierWriter(instance: BaggingClassifier)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      BaggingClassifierParams.saveImpl(instance, path, sc)
    }

  }

  private class BaggingClassifierReader extends MLReader[BaggingClassifier] {

    /** Checked against metadata when loading model */
    private val className = classOf[BaggingClassifier].getName

    override def load(path: String): BaggingClassifier = {
      val (metadata, learner) = BaggingClassifierParams.loadImpl(path, sc, className)
      val bc = new BaggingClassifier(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

class BaggingClassificationModel(
    override val uid: String,
    override val numClasses: Int,
    val subspaces: Array[Array[Int]],
    val models: Array[EnsemblePredictionModelType])
    extends ProbabilisticClassificationModel[Vector, BaggingClassificationModel]
    with BaggingClassifierParams
    with MLWritable {

  def this(
      numClasses: Int,
      subspaces: Array[Array[Int]],
      models: Array[EnsemblePredictionModelType]) =
    this(Identifiable.randomUID("BaggingClassificationModel"), numClasses, subspaces, models)

  val numModels: Int = models.length

  override def predictRaw(features: Vector): Vector = {
    val rawPredictions = Vectors.zeros(numClasses)
    var i = 0
    while (i < numModels) {
      val model = models(i)
      val subspace = subspaces(i)
      val rawPrediction = model match {
        case model: ProbabilisticClassificationModel[Vector, _]
            if ($(votingStrategy) == "soft") =>
          model.predictProbability(slice(subspace)(features))
        case model: ClassificationModel[Vector, _] if ($(votingStrategy) == "hard") =>
          Vectors.sparse(
            numClasses,
            Array(model.predict(slice(subspace)(features)).toInt),
            Array(1.0))
        case _ =>
          throw new SparkException(s"""voting strategy "${$(
              votingStrategy)}" is not compatible with base learner "${$(baseLearner)}".""")
      }
      BLAS.axpy(1.0, rawPrediction, rawPredictions)
      i += 1
    }
    rawPredictions
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    BLAS.scal(1.0 / numModels.toDouble, rawPrediction); rawPrediction
  }

  override def copy(extra: ParamMap): BaggingClassificationModel = {
    val copied = new BaggingClassificationModel(uid, numClasses, subspaces, models)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter =
    new BaggingClassificationModel.BaggingClassificationModelWriter(this)

}

object BaggingClassificationModel extends MLReadable[BaggingClassificationModel] {

  override def read: MLReader[BaggingClassificationModel] = new BaggingClassificationModelReader

  override def load(path: String): BaggingClassificationModel = super.load(path)

  private[BaggingClassificationModel] class BaggingClassificationModelWriter(
      instance: BaggingClassificationModel)
      extends MLWriter {

    private case class Data(subspace: Array[Int])

    override protected def saveImpl(path: String): Unit = {
      val extraJson =
        ("numClasses" -> instance.numClasses) ~ ("numModels" -> instance.numModels)
      BaggingClassifierParams.saveImpl(instance, path, sc, Some(extraJson))
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

  private class BaggingClassificationModelReader extends MLReader[BaggingClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BaggingClassificationModel].getName

    override def load(path: String): BaggingClassificationModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, baseLearner) = BaggingClassifierParams.loadImpl(path, sc, className)
      val numModels = (metadata.metadata \ "numModels").extract[Int]
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
      }
      val subspaces = (0 until numModels).toArray.map { idx =>
        val dataPath = new Path(path, s"data-$idx").toString
        val data = sparkSession.read.json(dataPath).select("subspace").head()
        data.getAs[Seq[Long]](0).map(_.toInt).toArray
      }
      val bcModel = new BaggingClassificationModel(metadata.uid, numClasses, subspaces, models)
      metadata.getAndSetParams(bcModel)
      bcModel.set("baseLearner", baseLearner)
      bcModel
    }
  }
}
