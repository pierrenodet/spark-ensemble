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

import java.util.{Locale, UUID}

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.boosting.BoostingParams
import org.apache.spark.ml.ensemble.{
  EnsemblePredictionModelType,
  EnsemblePredictorType,
  HasBaseLearner
}
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.storage.StorageLevel
import org.json4s.JsonAST.{JInt, JString, JValue}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{parse, render}
import org.json4s.{DefaultFormats, JObject, JsonAST}
import scala.util.Try

private[ml] trait BoostingRegressorParams extends BoostingParams {

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
        BoostingRegressorParams.supportedLossTypes.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getLoss: String = $(loss).toLowerCase(Locale.ROOT)

  setDefault(loss -> "exponential")

}

private[ml] object BoostingRegressorParams {

  final val supportedLossTypes: Array[String] =
    Array("exponential", "squared", "absolute").map(_.toLowerCase(Locale.ROOT))

  def lossFunction(loss: String): Double => Double = loss match {
    case "exponential" =>
      error => 1 - breeze.numerics.exp(-error)
    case "absolute" =>
      error => error
    case "squared" =>
      error => breeze.numerics.pow(error, 2)
    case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")

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
      expectedClassName: String): (DefaultParamsReader.Metadata, EnsemblePredictorType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val learner = HasBaseLearner.loadImpl(path, sc)
    (metadata, learner)
  }

}

class BoostingRegressor(override val uid: String)
    extends Predictor[Vector, BoostingRegressor, BoostingRegressionModel]
    with BoostingRegressorParams
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

  override def copy(extra: ParamMap): BoostingRegressor = {
    val copied = new BoostingRegressor(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }
  override protected def train(dataset: Dataset[_]): BoostingRegressionModel = instrumented {
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

      def trainBoosters(
          train: DataFrame,
          validation: DataFrame,
          labelColName: String,
          weightColName: Option[String],
          featuresColName: String,
          predictionColName: String,
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

          val errorColName = "boost$error" + UUID.randomUUID().toString
          val errors = booster
            .transform(probabilized)
            .withColumn(errorColName, abs(col(labelColName) - col(predictionColName)))
          val maxError = errors.agg(max(errorColName)).first().getDouble(0)

          val lossUDF = udf(loss)
          val lossColName = "boost$loss" + UUID.randomUUID().toString
          val losses = errors
            .withColumn(lossColName, coalesce(lossUDF(col(errorColName) / maxError), lit(0.0)))

          val avgl = avgLoss(lossColName, boostProbaColName)(losses)

          val b = beta(avgl)

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
              instrumentation)

          trainBoosters(
            selectedTrain,
            validation,
            labelColName,
            weightColName,
            featuresColName,
            predictionColName,
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
          boostWeightColName,
          withValidation,
          getBaseLearner,
          getNumBaseLearners,
          BoostingRegressorParams.lossFunction(getLoss),
          getTol,
          getNumRound,
          getSeed,
          instr)(Array.empty, Array.empty, getNumBaseLearners, Double.MaxValue, 0)

      if (handlePersistence) {
        train.unpersist()
        validation.unpersist()
      }

      new BoostingRegressionModel(weights, models)

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

  /* Here is the implementation for weighted median, but the weighted mean works better
  private val magicWeights = weights.map(weight => scala.math.log(1 / weight))
  private val magic: Double =
    0.5 * magicWeights.sum

  //Here we are using weighted mean, but in the paper the weighted median is preferred.
  override def predict(features: Vector): Double = {
    val sorted = models.map(_.predict(features)).zip(magicWeights).sortBy(_._1).map(_._2)
    println(sorted.mkString(","))
    val t = sorted.scanLeft(0.0)(_ + _).tail.map(_ >= magic).indexOf(true)
    println(t)
    models(t).predict(features)
  }
   */

  val numBaseModels: Int = models.length

  private val sumWeights: Double = weights.sum

  override def predict(features: Vector): Double = {
    BLAS.dot(Vectors.dense(models.map(_.predict(features))), Vectors.dense(weights)) / sumWeights
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
        Some("numBaseModels" -> instance.numBaseModels))
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

  private class BoostingRegressionModelReader extends MLReader[BoostingRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BoostingRegressionModel].getName

    override def load(path: String): BoostingRegressionModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = BoostingRegressorParams.loadImpl(path, sc, className)
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
      val bcModel =
        new BoostingRegressionModel(metadata.uid, boostsData, models)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
