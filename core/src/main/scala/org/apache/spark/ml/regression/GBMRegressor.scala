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

import breeze.numerics._
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.boosting.GBMParams
import org.apache.spark.ml.ensemble.HasSubBag.SubSpace
import org.apache.spark.ml.ensemble.{
  EnsemblePredictionModelType,
  EnsemblePredictorType,
  HasBaseLearner
}
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, ParamPair}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.sql.functions.{col, not, udf}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.storage.StorageLevel
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{parse, render}
import org.json4s.{DefaultFormats, JObject}

import scala.annotation.tailrec

private[ml] trait GBMRegressorParams extends GBMParams {

  /**
   * Loss function which Boosting tries to minimize. (case-insensitive)
   * Supported: "squared", "absolute", "huber", "quantile".
   * (default = ls)
   *
   * @group param
   */
  val loss: Param[String] =
    new Param(
      this,
      "loss",
      "loss function, (case-insensitive). Supported options:" + s"${GBMRegressorParams.supportedLossTypes
        .mkString(",")}",
      (value: String) =>
        GBMRegressorParams.supportedLossTypes.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getLoss: String = $(loss).toLowerCase(Locale.ROOT)

  setDefault(loss -> "squared")

  /**
   * The alpha-quantile of the huber loss function and the quantile loss function. Only if loss="huber" or loss="quantile".
   * (default = 0.9)
   *
   * @group param
   */
  val alpha: Param[Double] =
    new DoubleParam(
      this,
      "alpha",
      "The alpha-quantile of the loss function. Only for huber and quantile loss.")

  /** @group getParam */
  def getAlpha: Double = $(alpha)

  setDefault(alpha -> 0.9)

}

private[ml] object GBMRegressorParams {

  final val supportedLossTypes: Array[String] =
    Array("squared", "absolute", "huber", "quantile").map(_.toLowerCase(Locale.ROOT))

  def lossFunction(loss: String, alpha: Double): (Double, Double) => Double =
    loss match {
      case "squared" =>
        (y, prediction) => pow(y - prediction, 2) / 2.0
      case "absolute" =>
        (y, prediction) => abs(y - prediction)
      case "huber" =>
        (y, prediction) => pow(alpha, 2) * (sqrt(1.0 + pow((y - prediction) / alpha, 2)) - 1.0)
      case "quantile" =>
        (y, prediction) =>
          if (prediction > y) (alpha - 1.0) * (y - prediction) else alpha * (y - prediction)
      case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")
    }

  def gradFunction(loss: String, alpha: Double): (Double, Double) => Double =
    loss match {
      case "squared" =>
        (y, prediction) => -(y - prediction)
      case "absolute" =>
        (y, prediction) => -signum(y - prediction)
      case "huber" =>
        (y, prediction) => -(y - prediction) / sqrt(1 + pow((y - prediction) / alpha, 2))
      case "quantile" =>
        (y, prediction) => if (prediction > y) -(alpha - 1.0) else -alpha
      case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")
    }

  def saveImpl(
      instance: GBMRegressorParams,
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

class GBMRegressor(override val uid: String)
    extends Predictor[Vector, GBMRegressor, GBMRegressionModel]
    with GBMRegressorParams
    with MLWritable {

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setBaseLearner(value: Predictor[_, _, _]): this.type =
    set(baseLearner, value.asInstanceOf[EnsemblePredictorType])

  /** @group setParam */
  def setNumBaseLearners(value: Int): this.type = set(numBaseLearners, value)

  /** @group setParam */
  def setLoss(value: String): this.type = set(loss, value)

  /** @group setParam */
  def setAlpha(value: Double): this.type = set(alpha, value)

  /** @group setParam */
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  /** @group setParam */
  def setOptimizedWeights(value: Boolean): this.type = set(optimizedWeights, value)

  /** @group setParam */
  def setValidationIndicatorCol(value: String): this.type = set(validationIndicatorCol, value)

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  def setTol(value: Double): this.type = set(tol, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  def this() = this(Identifiable.randomUID("GBMRegressor"))

  override def copy(extra: ParamMap): GBMRegressor = {
    val copied = new GBMRegressor(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  override protected def train(dataset: Dataset[_]): GBMRegressionModel =
    instrumented { instr =>
      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(
        this,
        labelCol,
        weightCol,
        featuresCol,
        predictionCol,
        loss,
        numBaseLearners,
        learningRate,
        optimizedWeights,
        validationIndicatorCol,
        sampleRatio,
        replacement,
        subspaceRatio,
        maxIter,
        tol,
        seed)

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

      val bagColName = "gbm$bag" + UUID.randomUUID().toString
      val bagged = train.transform(
        withBag(getReplacement, getSampleRatio, getNumBaseLearners, getSeed, bagColName))

      val numFeatures = getNumFeatures(train, getFeaturesCol)

      @tailrec
      def trainBoosters(
          train: DataFrame,
          validation: DataFrame,
          labelColName: String,
          weightColName: Option[String],
          featuresColName: String,
          predictionColName: String,
          bagColName: String,
          withValidation: Boolean,
          baseLearner: EnsemblePredictorType,
          numBaseLearners: Int,
          learningRate: Double,
          loss: (Double, Double) => Double,
          grad: (Double, Double) => Double,
          sampleFeatureRatio: Double,
          numFeatures: Int,
          optimizedWeights: Boolean,
          maxIter: Int,
          tol: Double,
          numRound: Int,
          seed: Long,
          instrumentation: Instrumentation)(
          weights: Array[Double],
          subspaces: Array[SubSpace],
          boosters: Array[EnsemblePredictionModelType],
          const: Double,
          iter: Int,
          error: Double,
          numTry: Int): (Array[Double], Array[SubSpace], Array[EnsemblePredictionModelType]) = {

        if (iter == 0) {

          instrumentation.logInfo(s"Learning of GBM finished.")
          (weights.dropRight(numTry), subspaces.dropRight(numTry), boosters.dropRight(numTry))

        } else {

          instrumentation.logNamedValue("iteration", numBaseLearners - iter)

          val gradUDF = udf[Double, Double, Double](grad(_, _))

          val currentPredictionColName = "gbm$current" + UUID.randomUUID().toString
          val current = new GBMRegressionModel(weights, subspaces, boosters, const)
            .setPredictionCol(currentPredictionColName)
            .setFeaturesCol(featuresColName)

          val subspace = mkSubspace(sampleFeatureRatio, numFeatures, seed)

          val residualsColName = "gbm$residuals" + UUID.randomUUID().toString
          val residuals = current
            .transform(train)
            .withColumn(
              residualsColName,
              -gradUDF(col(labelColName), col(currentPredictionColName)))

          val subbag = residuals.transform(
            extractSubBag(bagColName, numBaseLearners - iter, featuresColName, subspace))

          val booster = fitBaseLearner(
            baseLearner,
            residualsColName,
            featuresColName,
            predictionColName,
            weightColName)(subbag)

          val weight = if (getOptimizedWeights) {

            val boosterPredictionColName = "gbm$booster" + UUID.randomUUID().toString
            val transformed =
              booster.setPredictionCol(boosterPredictionColName).transform(subbag)

            learningRate * findOptimizedWeight(
              labelColName,
              currentPredictionColName,
              boosterPredictionColName,
              loss,
              grad,
              maxIter,
              tol)(transformed)

          } else {

            learningRate * 1.0

          }

          instrumentation.logNamedValue("weight", weight)

          val updatedWeights = weights :+ weight
          val updatedBoosters = boosters :+ booster
          val updatedSubspaces = subspaces :+ subspace

          val updatedModel =
            new GBMRegressionModel(updatedWeights, updatedSubspaces, updatedBoosters, const)

          val verror =
            evaluateOnValidation(updatedModel, labelColName, loss)(validation)

          val (updatedIter, updatedError, updatedNumTry) =
            terminate(
              weight,
              learningRate,
              withValidation,
              error,
              verror,
              tol,
              numRound,
              numTry,
              iter,
              instrumentation)

          trainBoosters(
            train,
            validation,
            labelColName,
            weightColName,
            featuresColName,
            predictionColName,
            bagColName,
            withValidation,
            baseLearner,
            numBaseLearners,
            learningRate,
            loss,
            grad,
            sampleFeatureRatio,
            numFeatures,
            optimizedWeights,
            maxIter,
            tol,
            numRound,
            seed + iter,
            instrumentation)(
            updatedWeights,
            updatedSubspaces,
            updatedBoosters,
            const,
            updatedIter,
            updatedError,
            updatedNumTry)

        }

      }

      val optWeightColName = if (weightColIsUsed) {
        Some($(weightCol))
      } else {
        None
      }

      val initConst = if (getOptimizedWeights) {
        findOptimizedConst(
          getLabelCol,
          GBMRegressorParams.lossFunction(getLoss, getAlpha),
          GBMRegressorParams.gradFunction(getLoss, getAlpha),
          getMaxIter,
          getTol)(train)
      } else {
        0.0
      }

      val (weights, subspaces, boosters) =
        trainBoosters(
          bagged,
          validation,
          getLabelCol,
          optWeightColName,
          getFeaturesCol,
          getPredictionCol,
          bagColName,
          withValidation,
          getBaseLearner,
          getNumBaseLearners,
          getLearningRate,
          GBMRegressorParams.lossFunction(getLoss, getAlpha),
          GBMRegressorParams.gradFunction(getLoss, getAlpha),
          getSubspaceRatio,
          numFeatures,
          getOptimizedWeights,
          getMaxIter,
          getTol,
          getNumRound,
          getSeed,
          instr)(
          Array.empty,
          Array.empty,
          Array.empty,
          initConst,
          getNumBaseLearners,
          Double.MaxValue,
          0)

      if (handlePersistence) {
        train.unpersist()
        validation.unpersist()
      }

      new GBMRegressionModel(weights, subspaces, boosters, initConst)

    }

  override def write: MLWriter =
    new GBMRegressor.GBMRegressorWriter(this)

}

object GBMRegressor extends MLReadable[GBMRegressor] {

  override def read: MLReader[GBMRegressor] = new GBMRegressorReader

  override def load(path: String): GBMRegressor = super.load(path)

  private[GBMRegressor] class GBMRegressorWriter(instance: GBMRegressor) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      GBMRegressorParams.saveImpl(instance, path, sc)
    }

  }

  private class GBMRegressorReader extends MLReader[GBMRegressor] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBMRegressor].getName

    override def load(path: String): GBMRegressor = {
      val (metadata, learner) = GBMRegressorParams.loadImpl(path, sc, className)
      val bc = new GBMRegressor(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

class GBMRegressionModel(
    override val uid: String,
    val weights: Array[Double],
    val subspaces: Array[SubSpace],
    val models: Array[EnsemblePredictionModelType],
    val const: Double)
    extends PredictionModel[Vector, GBMRegressionModel]
    with GBMRegressorParams
    with MLWritable {

  def this(
      weights: Array[Double],
      subspaces: Array[SubSpace],
      models: Array[EnsemblePredictionModelType],
      const: Double) =
    this(Identifiable.randomUID("BoostingRegressionModel"), weights, subspaces, models, const)

  val numBaseModels: Int = models.length

  override def predict(features: Vector): Double = {
    BLAS.dot(Vectors.dense(models.zip(subspaces).map {
      case (model, subspace) =>
        model.predict(slicer(subspace)(features))
    }), Vectors.dense(weights)) + const
  }

  override def copy(extra: ParamMap): GBMRegressionModel = {
    val copied = new GBMRegressionModel(uid, weights, subspaces, models, const)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter =
    new GBMRegressionModel.GBMRegressionModelWriter(this)

}

object GBMRegressionModel extends MLReadable[GBMRegressionModel] {

  override def read: MLReader[GBMRegressionModel] =
    new GBMRegressionModelReader

  override def load(path: String): GBMRegressionModel = super.load(path)

  private[GBMRegressionModel] class GBMRegressionModelWriter(instance: GBMRegressionModel)
      extends MLWriter {

    private case class Data(weight: Double, subspace: SubSpace, const: Double)

    override protected def saveImpl(path: String): Unit = {
      GBMRegressorParams.saveImpl(
        instance,
        path,
        sc,
        Some("numBaseModels" -> instance.numBaseModels))
      instance.models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach {
        case (model, idx) =>
          val modelPath = new Path(path, s"model-$idx").toString
          model.save(modelPath)
      }
      instance.weights.zip(instance.subspaces).zipWithIndex.foreach {
        case ((weight, subspace), idx) =>
          val data = Data(weight, subspace, instance.const)
          val dataPath = new Path(path, s"data-$idx").toString
          sparkSession.createDataFrame(Seq(data)).repartition(1).write.json(dataPath)
      }

    }
  }

  private class GBMRegressionModelReader extends MLReader[GBMRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBMRegressionModel].getName

    override def load(path: String): GBMRegressionModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = GBMRegressorParams.loadImpl(path, sc, className)
      val numModels = (metadata.metadata \ "numBaseModels").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
      }
      val boostsData = (0 until numModels).toArray.map { idx =>
        val dataPath = new Path(path, s"data-$idx").toString
        val data = sparkSession.read.json(dataPath).select("weight", "subspace", "const").head()
        (
          data.getAs[Double](0),
          data.getAs[Seq[Long]](1).map(_.toInt).toArray,
          data.getAs[Double](2))
      }.unzip3
      val bcModel =
        new GBMRegressionModel(
          metadata.uid,
          boostsData._1,
          boostsData._2,
          models,
          boostsData._3(0))
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
