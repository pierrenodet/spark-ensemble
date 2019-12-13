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
import java.util.{Locale, UUID}

import breeze.linalg.DenseVector
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.boosting.BoostingParams
import org.apache.spark.ml.ensemble.{
  EnsemblePredictionModelType,
  EnsemblePredictorType,
  HasBaseLearner
}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{parse, render}
import org.json4s.{DefaultFormats, JObject}

private[ml] trait BoostingClassifierParams extends BoostingParams with ClassifierParams {

  /**
   * Loss function which Boosting tries to minimize. (case-insensitive)
   * Supported: "exponential"
   * (default = exponential)
   *
   * @group param
   */
  val loss: Param[String] =
    new Param(
      this,
      "loss",
      "loss function, exponential by default",
      (value: String) =>
        BoostingClassifierParams.supportedLossTypes.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getLoss: String = $(loss).toLowerCase(Locale.ROOT)

  setDefault(loss -> "exponential")

}

private[ml] object BoostingClassifierParams {

  final val supportedLossTypes: Array[String] =
    Array("exponential").map(_.toLowerCase(Locale.ROOT))

  def lossFunction(loss: String, numClasses: Int): Double => Double = loss match {
    case "exponential" =>
      error => 1 - breeze.numerics.exp(-error)
    case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")

  }

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
      expectedClassName: String): (DefaultParamsReader.Metadata, EnsemblePredictorType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val learner = HasBaseLearner.loadImpl(path, sc)
    (metadata, learner)
  }

}

class BoostingClassifier(override val uid: String)
    extends Classifier[Vector, BoostingClassifier, BoostingClassificationModel]
    with BoostingClassifierParams
    with MLWritable {

  def setBaseLearner(value: Predictor[_, _, _]): this.type =
    set(baseLearner, value.asInstanceOf[EnsemblePredictorType])

  /** @group setParam */
  def setNumBaseLearners(value: Int): this.type = set(numBaseLearners, value)

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  def setLoss(value: String): this.type = set(loss, value)

  /** @group setParam */
  def setValidationIndicatorCol(value: String): this.type = set(validationIndicatorCol, value)

  /** @group setParam */
  def setTol(value: Double): this.type = set(tol, value)

  /** @group setParam */
  def setNumRound(value: Int): this.type = set(numRound, value)

  def this() = this(Identifiable.randomUID("BoostingRegressor"))

  override def copy(extra: ParamMap): BoostingClassifier = {
    val copied = new BoostingClassifier(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  /**
   * Validates that number of classes is greater than zero.
   *
   * @param numClasses Number of classes label can take.
   */
  protected def validateNumClasses(numClasses: Int): Unit = {
    require(
      numClasses > 0,
      s"Classifier (in extractLabeledPoints) found numClasses =" +
        s" $numClasses, but requires numClasses > 0.")
  }

  /**
   * Validates the label on the classifier is a valid integer in the range [0, numClasses).
   *
   * @param label The label to validate.
   * @param numClasses Number of classes label can take.  Labels must be integers in the range
   *                  [0, numClasses).
   */
  protected def validateLabel(label: Double, numClasses: Int): Unit = {
    require(
      label.toLong == label && label >= 0 && label < numClasses,
      s"Classifier was given" +
        s" dataset with invalid label $label.  Labels must be integers in range" +
        s" [0, $numClasses).")
  }

  override protected def train(dataset: Dataset[_]): BoostingClassificationModel = instrumented {
    instr =>
      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(this, numBaseLearners, seed)

      val weightColIsUsed = isDefined(weightCol) && $(weightCol).nonEmpty && {
        getBaseLearner match {
          case _: HasWeightCol => true
          case c =>
            instr.logWarning(s"weightCol is ignored, as it is not supported by $c now.")
            false
        }
      }

      val withValidation = isDefined(validationIndicatorCol) && $(validationIndicatorCol).nonEmpty

      val df = if (weightColIsUsed) {
        dataset.select($(labelCol), $(weightCol), $(featuresCol))
      } else {
        dataset.select($(labelCol), $(featuresCol))
      }

      val optWeightColName = if (weightColIsUsed) {
        Some($(weightCol))
      } else {
        None
      }

      val (train, validation) = if (withValidation) {
        (
          df.filter(not(col($(validationIndicatorCol)))),
          df.filter(col($(validationIndicatorCol))))
      } else {
        (df, df.sparkSession.emptyDataFrame)
      }

      val handlePersistence = dataset.storageLevel == StorageLevel.NONE && (train.storageLevel == StorageLevel.NONE) && (validation.storageLevel == StorageLevel.NONE)
      if (handlePersistence) {
        train.persist(StorageLevel.MEMORY_AND_DISK)
        validation.persist(StorageLevel.MEMORY_AND_DISK)
      }

      val boostWeightColName = "boost$weight" + UUID.randomUUID().toString
      val weighted = train.withColumn(boostWeightColName, lit(1.0))

      val labelSchema = dataset.schema($(labelCol))
      val computeNumClasses: () => Int = () => {
        val Row(maxLabelIndex: Double) =
          dataset.agg(max(col($(labelCol)).cast(DoubleType))).head()
        // classes are assumed to be numbered from 0,...,maxLabelIndex
        maxLabelIndex.toInt + 1
      }
      val numClasses =
        MetadataUtils.getNumClasses(labelSchema).fold(computeNumClasses())(identity)
      instr.logNumClasses(numClasses)

      validateNumClasses(numClasses)

      def trainBoosters(
          train: DataFrame,
          validation: DataFrame,
          labelColName: String,
          weightColName: Option[String],
          featuresColName: String,
          predictionColName: String,
          rawPredictionColName: String,
          numClasses: Int,
          boostWeightColName: String,
          withValidation: Boolean,
          baseLearner: EnsemblePredictorType,
          numBaseLearners: Int,
          loss: Double => Double,
          tol: Double,
          numRound: Int,
          seed: Long,
          instrumentation: Instrumentation)(
          weights: Array[Double],
          boosters: Array[EnsemblePredictionModelType],
          iter: Int,
          error: Double,
          numTry: Int): (Array[Double], Array[EnsemblePredictionModelType]) = {

        if (iter == 0) {
          instrumentation.logInfo(s"Learning of Boosting finished.")
          (weights.dropRight(numTry), boosters.dropRight(numTry))
        } else {

          instrumentation.logNamedValue("iteration", numBaseLearners - iter)

          val boostProbaColName = "boost$proba" + UUID.randomUUID().toString
          val poissonProbaColName = "poisson$proba" + UUID.randomUUID().toString

          val probabilized =
            probabilize(boostWeightColName, boostProbaColName, poissonProbaColName)(train)

          val replicated = extractBoostedBag(poissonProbaColName, seed)(probabilized)

          val booster = fitBaseLearner(
            baseLearner,
            labelColName,
            featuresColName,
            predictionColName,
            weightColName)(replicated)

          val lossColName = "boost$loss" + UUID.randomUUID().toString
          val lossUDF = udf(loss)
          val losses = booster
            .transform(probabilized)
            .withColumn(
              lossColName,
              lossUDF(when(col(labelColName) === col(predictionColName), 0.0).otherwise(1.0)))

          val avgl = avgLoss(lossColName, boostProbaColName)(losses)

          val b = beta(avgl, numClasses)

          val w = weight(b)

          val updatedBoostWeightColName = "boost$weight" + UUID.randomUUID().toString
          val updatedTrain =
            updateWeights(boostWeightColName, lossColName, b, updatedBoostWeightColName)(losses)

          val selectedTrain = updatedTrain.select(
            (train.columns.filter(_ != boostWeightColName) :+ updatedBoostWeightColName)
              .map(col): _*)

          val updatedWeights = weights :+ w
          val updatedBoosters = boosters :+ booster

          val verror = evaluateOnValidation(
            numClasses,
            updatedWeights,
            updatedBoosters,
            labelColName,
            featuresColName,
            loss)(validation)
          instrumentation.logNamedValue("Error on Validation", verror)

          val (updatedIter, updatedError, updatedNumTry) =
            terminate(
              avgl,
              withValidation,
              error,
              verror,
              tol,
              numRound,
              numTry,
              iter,
              instrumentation,
              numClasses)

          trainBoosters(
            selectedTrain,
            validation,
            labelColName,
            weightColName,
            featuresColName,
            predictionColName,
            rawPredictionColName,
            numClasses,
            updatedBoostWeightColName,
            withValidation,
            baseLearner,
            numBaseLearners,
            loss,
            tol,
            numRound,
            seed + iter,
            instrumentation)(
            updatedWeights,
            updatedBoosters,
            updatedIter,
            updatedError,
            updatedNumTry)
        }
      }

      val (weights, models) =
        trainBoosters(
          weighted,
          validation,
          getLabelCol,
          optWeightColName,
          getFeaturesCol,
          getPredictionCol,
          getRawPredictionCol,
          numClasses,
          boostWeightColName,
          withValidation,
          getBaseLearner,
          getNumBaseLearners,
          BoostingClassifierParams.lossFunction(getLoss, numClasses),
          getTol,
          getNumRound,
          getSeed,
          instr)(Array.empty, Array.empty, getNumBaseLearners, Double.MaxValue, 0)

      if (handlePersistence) {
        train.unpersist()
        validation.unpersist()
      }

      new BoostingClassificationModel(numClasses, weights, models)

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
    extends ClassificationModel[Vector, BoostingClassificationModel]
    with BoostingClassifierParams
    with MLWritable {

  def this(numClasses: Int, weights: Array[Double], models: Array[EnsemblePredictionModelType]) =
    this(Identifiable.randomUID("BoostingRegressionModel"), numClasses, weights, models)

  val numBaseModels: Int = models.length

  override protected def predictRaw(features: Vector): Vector = {
    val tmp = DenseVector.zeros[Double](numClasses)
    weights
      .zip(models)
      .foreach {
        case (weight, model) =>
          tmp(model.predict(features).ceil.toInt) += 1.0 * weight
      }
    Vectors.fromBreeze(tmp)
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
        ("numClasses" -> instance.numClasses) ~ ("numBaseModels" -> instance.numBaseModels)
      BoostingClassifierParams.saveImpl(instance, path, sc, Some(extraJson))
      instance.models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach {
        case (model, idx) =>
          val modelPath = new Path(path, s"model-$idx").toString
          model.save(modelPath)
      }
      instance.weights.zipWithIndex.foreach {
        case (weight, idx) =>
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
      val numModels = (metadata.metadata \ "numBaseModels").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
      }
      val boostsData = (0 until numModels).toArray.map { idx =>
        val dataPath = new Path(path, s"data-$idx").toString
        val data = sparkSession.read.json(dataPath).select("weight").head()
        data.getAs[Double](0)
      }
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val bcModel =
        new BoostingClassificationModel(metadata.uid, numClasses, boostsData, models)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
