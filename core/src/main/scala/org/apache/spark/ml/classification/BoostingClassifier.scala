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

import java.util.Locale

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.boosting.BoostingParams
import org.apache.spark.ml.ensemble.{
  EnsemblePredictionModelType,
  HasBaseLearner,
  EnsembleClassifierType
}
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql.Dataset
import org.apache.spark.storage.StorageLevel
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{parse, render}
import org.json4s.{DefaultFormats, JObject}
import org.apache.spark.rdd.util.PeriodicRDDCheckpointer
import org.apache.spark.ml.impl.Utils.softmax
import org.apache.spark.ml.util.MLReader
import org.apache.spark.SparkException
import org.apache.spark.ml.impl.Utils.EPSILON
import org.apache.spark.ml.ensemble.Utils

private[ml] trait BoostingClassifierParams extends BoostingParams[EnsembleClassifierType] {

  /**
   * Discrete (SAMME) or Real (SAMME.R) boosting algorithm. (case-insensitive) Supported:
   * "discrete", "real". (default = median)
   *
   * @group param
   */
  val algorithm: Param[String] =
    new Param(
      this,
      "algorithm",
      "algorithm, (case-insensitive). Supported options:" + s"${BoostingClassifierParams.supportedAlgorithm
          .mkString(",")}",
      (value: String) =>
        BoostingClassifierParams.supportedAlgorithm.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getAlgorithm: String = $(algorithm).toLowerCase(Locale.ROOT)

  setDefault(algorithm -> "discrete")

}

private[ml] object BoostingClassifierParams {

  final val supportedAlgorithm: Array[String] =
    Array("discrete", "real").map(_.toLowerCase(Locale.ROOT))

  def saveImpl(
      instance: BoostingClassifierParams,
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

class BoostingClassifier(override val uid: String)
    extends Classifier[Vector, BoostingClassifier, BoostingClassificationModel]
    with BoostingClassifierParams
    with MLWritable {

  def setBaseLearner(value: EnsembleClassifierType): this.type =
    set(baseLearner, value)

  /** @group setParam */
  def setNumBaseLearners(value: Int): this.type = set(numBaseLearners, value)

  /** @group setParam */
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /** @group setParam */
  def setAlgorithm(value: String): this.type = set(algorithm, value)

  def this() = this(Identifiable.randomUID("BoostingClassifier"))

  override def copy(extra: ParamMap): BoostingClassifier = {
    val copied = new BoostingClassifier(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  def error(label: Double, prediction: Double): Double = if (label != prediction) 1.0 else 0.0

  override protected def train(dataset: Dataset[_]): BoostingClassificationModel = instrumented {
    instr =>
      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(
        this,
        labelCol,
        featuresCol,
        predictionCol,
        weightCol,
        algorithm,
        numBaseLearners,
        checkpointInterval)

      val spark = dataset.sparkSession
      val sc = spark.sparkContext

      val numClasses = getNumClasses(dataset)
      instr.logNumClasses(numClasses)
      validateNumClasses(numClasses)

      val instances =
        extractInstances(dataset, instance => validateLabel(instance.label, numClasses))

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

        model match {
          case model: ProbabilisticClassificationModel[Vector, _] if ($(algorithm) == "real") => {
            val probabilities =
              weighted.map(instance => model.predictProbability(instance.features))

            val estimatorError = weighted
              .zip(probabilities)
              .treeAggregate(0.0)(
                { case (acc, (instance, probability)) =>
                  acc + instance.weight * error(instance.label, probability.argmax)
                },
                { _ + _ },
                $(aggregationDepth))
            if (estimatorError <= 0) done = true

            estimatorWeights(i) = 1.0
            models(i) = model

            boostingWeights = weighted
              .zip(probabilities)
              .map {
                case (instance, probability) => {
                  var loss = 0.0
                  var i = 0
                  while (i < numClasses) {
                    val code = if (instance.label == i) 1 else -1 / (numClasses - 1.0)
                    loss += code * math.log(math.max(probability(i), EPSILON))
                    i += 1
                  }
                  instance.weight * math.exp(-((numClasses - 1.0) / numClasses) * loss)
                }
              }

          }
          case model: ClassificationModel[Vector, _] if ($(algorithm) == "discrete") => {
            val errors =
              weighted.map(instance => error(instance.label, model.predict(instance.features)))

            val estimatorError = weighted
              .zip(errors)
              .treeAggregate(0d)(
                { case (acc, (instance, error)) =>
                  acc + instance.weight * error
                },
                { _ + _ },
                $(aggregationDepth))

            if (estimatorError <= 0) done = true

            val beta = estimatorError / ((1 - estimatorError) * (numClasses - 1))
            val estimatorWeight = if (beta == 0.0) 1.0 else math.log(1.0 / beta)

            estimatorWeights(i) = estimatorWeight
            models(i) = model

            if (estimatorError >= 1.0 - (1.0 / numClasses)) { i = i - 1; done = true }

            boostingWeights = weighted
              .zip(errors)
              .map { case (instance, error) =>
                instance.weight * math.pow(1 / beta, error)
              }

          }
          case _ =>
            throw new SparkException(s"""algorithm "${$(
                algorithm)}" is not compatible with base learner "${$(baseLearner)}".""")

        }

        boostingWeightsCheckpointer.update(boostingWeights)

        sumWeights = boostingWeights.treeReduce((_ + _), $(aggregationDepth))

        i += 1

      }

      boostingWeightsCheckpointer.unpersistDataSet()
      boostingWeightsCheckpointer.deleteAllCheckpoints()

      instances.unpersist()

      new BoostingClassificationModel(numClasses, estimatorWeights.take(i), models.take(i))

  }

  override def write: MLWriter = new BoostingClassifier.BoostingClassifierWriter(this)

}

object BoostingClassifier extends MLReadable[BoostingClassifier] {

  override def read: MLReader[BoostingClassifier] = new BoostingClassifierReader

  override def load(path: String): BoostingClassifier = super.load(path)

  private[BoostingClassifier] class BoostingClassifierWriter(instance: BoostingClassifier)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      BoostingClassifierParams.saveImpl(instance, path, sc)
    }

  }

  private class BoostingClassifierReader extends MLReader[BoostingClassifier] {

    /** Checked against metadata when loading model */
    private val className = classOf[BoostingClassifier].getName

    override def load(path: String): BoostingClassifier = {
      val (metadata, learner) = BoostingClassifierParams.loadImpl(path, sc, className)
      val bc = new BoostingClassifier(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

class BoostingClassificationModel(
    override val uid: String,
    override val numClasses: Int,
    val weights: Array[Double],
    val models: Array[EnsemblePredictionModelType])
    extends ProbabilisticClassificationModel[Vector, BoostingClassificationModel]
    with BoostingClassifierParams
    with MLWritable {

  def this(numClasses: Int, weights: Array[Double], models: Array[EnsemblePredictionModelType]) =
    this(Identifiable.randomUID("BoostingClassificationModel"), numClasses, weights, models)

  val numModels = models.size

  weights.sum

  override def predictRaw(features: Vector): Vector = {
    val rawPredictions = $(algorithm) match {
      case "real" => predictRawReal(features)
      case "discrete" => predictRawDiscrete(features)
    }
    rawPredictions
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    BLAS.scal(1.0 / (numClasses - 1.0), rawPrediction)
    softmax(rawPrediction.toArray)
    rawPrediction
  }

  private def predictRawReal(features: Vector): Vector = {
    val res = Vectors.zeros(numClasses)
    var i = 0
    while (i < numModels) {
      val probabilisticModel = models(i).asInstanceOf[ProbabilisticClassificationModel[Vector, _]]
      val logProbabilities = probabilisticModel
        .predictProbability(features)
        .toArray
        .map(probability => math.log(math.max(probability, EPSILON)))
      val sumLogProba = logProbabilities.sum
      val decisions =
        Vectors.dense(logProbabilities.map(_ - (1.0 / numClasses) * sumLogProba))
      BLAS.axpy(numClasses - 1, decisions, res)
      i += 1
    }
    res
  }

  private def predictRawDiscrete(features: Vector): Vector = {
    val res = Array.fill(numClasses)(0.0)
    var i = 0
    while (i < numModels) {
      res(models(i).predict(features).toInt) += weights(i)
      i += 1
    }
    Vectors.dense(res)
  }

  override def copy(extra: ParamMap): BoostingClassificationModel = {
    val copied = new BoostingClassificationModel(uid, numClasses, weights, models)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter =
    new BoostingClassificationModel.BoostingClassificationModelWriter(this)

}

object BoostingClassificationModel extends MLReadable[BoostingClassificationModel] {

  override def read: MLReader[BoostingClassificationModel] = new BoostingClassificationModelReader

  override def load(path: String): BoostingClassificationModel = super.load(path)

  private[BoostingClassificationModel] class BoostingClassificationModelWriter(
      instance: BoostingClassificationModel)
      extends MLWriter {

    private case class Data(weight: Double)

    override protected def saveImpl(path: String): Unit = {
      val extraJson =
        ("numClasses" -> instance.numClasses) ~ ("numModels" -> instance.numModels)
      BoostingClassifierParams.saveImpl(instance, path, sc, Some(extraJson))
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

  private class BoostingClassificationModelReader extends MLReader[BoostingClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BoostingClassificationModel].getName

    override def load(path: String): BoostingClassificationModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = BoostingClassifierParams.loadImpl(path, sc, className)
      val numModels = (metadata.metadata \ "numModels").extract[Int]
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]

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
        new BoostingClassificationModel(metadata.uid, numClasses, boostsData, models)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
