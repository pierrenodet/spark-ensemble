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

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.boosting.GBMParams
import org.apache.spark.ml.ensemble.HasSubBag.SubSpace
import org.apache.spark.ml.ensemble._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.shared.{HasParallelism, HasWeightCol}
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.ThreadUtils
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{parse, render}
import org.json4s.{DefaultFormats, JObject}

import scala.annotation.tailrec
import scala.concurrent.duration.Duration
import scala.concurrent.{ExecutionContext, Future}

private[ml] trait GBMClassifierParams
    extends ClassifierParams
    with GBMParams
    with HasParallelism {

  /**
   * Loss function which Boosting tries to minimize. (case-insensitive)
   * Supported: "divergence".
   * (default = lk)
   *
   * @group param
   */
  val loss: Param[String] =
    new Param(
      this,
      "loss",
      "loss function, (case-insensitive). Supported options:" + s"${GBMClassifierParams.supportedLossTypes
        .mkString(",")}",
      (value: String) =>
        GBMClassifierParams.supportedLossTypes.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getLoss: String = $(loss).toLowerCase(Locale.ROOT)

  setDefault(loss -> "divergence")

}

private[ml] object GBMClassifierParams {

  final val supportedLossTypes: Array[String] =
    Array("divergence").map(_.toLowerCase(Locale.ROOT))

  def lossFunction(loss: String): (Double, Double) => Double = loss match {
    case "divergence" =>
      (y, prediction) => -y * breeze.numerics.log(prediction)
    case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")
  }

  def gradFunction(loss: String): (Double, Double) => Double = loss match {
    case "divergence" =>
      (y, prediction) => -(y - prediction)
    case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")
  }

  def saveImpl(
      instance: GBMClassifierParams,
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

class GBMClassifier(override val uid: String)
    extends Classifier[Vector, GBMClassifier, GBMClassificationModel]
    with GBMClassifierParams
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

  def this() = this(Identifiable.randomUID("GBMClassifier"))

  override def copy(extra: ParamMap): GBMClassifier = {
    val copied = new GBMClassifier(uid)
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

  override protected def train(dataset: Dataset[_]): GBMClassificationModel =
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
        maxIter,
        learningRate,
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
        dataset.select($(labelCol), $(featuresCol), $(weightCol))
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

      dataset
        .select(col($(labelCol)))
        .rdd
        .map { case Row(label: Double) => label }
        .foreach(validateLabel(_, numClasses))

      @tailrec
      def trainBoosters(
          train: DataFrame,
          validation: DataFrame,
          labelColName: String,
          weightColName: Option[String],
          featuresColName: String,
          predictionColName: String,
          rawPredictionColName: String,
          numClasses: Int,
          executionContext: ExecutionContext,
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
          weights: Array[Array[Double]],
          subspaces: Array[SubSpace],
          boosters: Array[Array[EnsemblePredictionModelType]],
          iter: Int,
          error: Double,
          numTry: Int)
          : (Array[Array[Double]], Array[SubSpace], Array[Array[EnsemblePredictionModelType]]) = {

        if (iter == 0) {

          instrumentation.logInfo(s"Learning of GBM finished.")
          (weights.dropRight(numTry), subspaces.dropRight(numTry), boosters.dropRight(numTry))

        } else {

          instrumentation.logNamedValue("iteration", numBaseLearners - iter)

          val gradUDF = udf[Double, Double, Double](grad(_, _))

          val currentPredictionColName = "gbm$current" + UUID.randomUUID().toString
          val currentRawPredictionColName = "gbm$current-raw" + UUID.randomUUID().toString
          val current = new GBMClassificationModel(numClasses, weights, subspaces, boosters)
            .setRawPredictionCol(currentRawPredictionColName)
            .setPredictionCol(currentPredictionColName)
            .setFeaturesCol(featuresColName)

          val subspace = mkSubspace(sampleFeatureRatio, numFeatures, seed)

          val weightedBoosterFutures = Array
            .range(0, numClasses)
            .map(k =>
              Future {

                val vecToArrUDF =
                  udf[Array[Double], Vector]((features: Vector) => features.toArray)

                val residualsColName = "gbm$residuals" + UUID.randomUUID().toString
                val relabeledColName = "gbm$relabeled" + UUID.randomUUID().toString
                val currentPartialPredColName = "gbm$current-partial" + UUID.randomUUID().toString

                val residuals = current
                  .transform(train)
                  .withColumn(
                    relabeledColName,
                    when(col(labelColName) === k.toDouble, 1.0).otherwise(0.0))
                  .withColumn(
                    currentPartialPredColName,
                    element_at(vecToArrUDF(col(currentRawPredictionColName)), k + 1))
                  .withColumn(
                    residualsColName,
                    -gradUDF(col(relabeledColName), col(currentPartialPredColName)))

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
                    relabeledColName,
                    currentPartialPredColName,
                    boosterPredictionColName,
                    loss,
                    grad,
                    maxIter,
                    tol)(transformed)

                } else {

                  learningRate * 1.0

                }

                (weight, booster)

              }(executionContext))

          val (weight, booster) = weightedBoosterFutures
            .map(ThreadUtils.awaitResult(_, Duration.Inf))
            .unzip[Double, EnsemblePredictionModelType]

          instrumentation.logNamedValue("weight", weight)

          val updatedWeights = weights :+ weight
          val updatedBoosters = boosters :+ booster
          val updatedSubspaces = subspaces :+ subspace

          val updatedModel = new GBMClassificationModel(
            numClasses,
            updatedWeights,
            updatedSubspaces,
            updatedBoosters)

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
            rawPredictionColName,
            numClasses,
            executionContext,
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
            numTry,
            seed,
            instrumentation)(
            updatedWeights,
            updatedSubspaces,
            updatedBoosters.asInstanceOf[Array[Array[EnsemblePredictionModelType]]],
            updatedIter,
            updatedError,
            updatedNumTry)
        }

      }

      val executionContext = getExecutionContext

      val optWeightColName = if (weightColIsUsed) {
        Some($(weightCol))
      } else {
        None
      }

      val (weights, subspaces, boosters) =
        trainBoosters(
          bagged,
          validation,
          getLabelCol,
          optWeightColName,
          getFeaturesCol,
          getPredictionCol,
          getRawPredictionCol,
          numClasses,
          executionContext,
          bagColName,
          withValidation,
          getBaseLearner,
          getNumBaseLearners,
          getLearningRate,
          GBMClassifierParams.lossFunction(getLoss),
          GBMClassifierParams.gradFunction(getLoss),
          getSubspaceRatio,
          numFeatures,
          getOptimizedWeights,
          getMaxIter,
          getTol,
          getNumRound,
          getSeed,
          instr)(Array.empty, Array.empty, Array.empty, getNumBaseLearners, Double.MaxValue, 0)

      if (handlePersistence) {
        df.unpersist()
      }

      new GBMClassificationModel(numClasses, weights, subspaces, boosters)

    }

  override def write: MLWriter =
    new GBMClassifier.GBMClassifierWriter(this)

}

object GBMClassifier extends MLReadable[GBMClassifier] {

  override def read: MLReader[GBMClassifier] = new GBMClassifierReader

  override def load(path: String): GBMClassifier = super.load(path)

  private[GBMClassifier] class GBMClassifierWriter(instance: GBMClassifier) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      GBMClassifierParams.saveImpl(instance, path, sc)
    }

  }

  private class GBMClassifierReader extends MLReader[GBMClassifier] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBMClassifier].getName

    override def load(path: String): GBMClassifier = {
      val (metadata, learner) = GBMClassifierParams.loadImpl(path, sc, className)
      val bc = new GBMClassifier(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

/* The models and weights are first indexed by the number of iterations then the number of classes, weights(0) contains k elements for the k classes*/
class GBMClassificationModel(
    override val uid: String,
    override val numClasses: Int,
    val weights: Array[Array[Double]],
    val subspaces: Array[SubSpace],
    val models: Array[Array[EnsemblePredictionModelType]])
    extends ClassificationModel[Vector, GBMClassificationModel]
    with GBMClassifierParams
    with MLWritable {

  def this(
      numClasses: Int,
      weights: Array[Array[Double]],
      subspaces: Array[SubSpace],
      models: Array[Array[EnsemblePredictionModelType]]) =
    this(Identifiable.randomUID("GBMClassificationModel"), numClasses, weights, subspaces, models)

  val numBaseModels: Int = models.length

  override def predictRaw(features: Vector): Vector = {

    val res = Vectors.zeros(numClasses).asBreeze

    Array.range(0, numBaseModels).foreach { m =>
      Array.range(0, numClasses).foreach { k =>
        res(k) += models(m)(k).predict(slicer(subspaces(m))(features)) * weights(m)(k)
      }
    }

    Vectors.fromBreeze(breeze.numerics.exp(res) / breeze.linalg.sum(breeze.numerics.exp(res)))

  }

  override def copy(extra: ParamMap): GBMClassificationModel = {
    val copied = new GBMClassificationModel(uid, numClasses, weights, subspaces, models)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter =
    new GBMClassificationModel.GBMClassificationModelWriter(this)

}

object GBMClassificationModel extends MLReadable[GBMClassificationModel] {

  override def read: MLReader[GBMClassificationModel] =
    new GBMClassificationModelReader

  override def load(path: String): GBMClassificationModel = super.load(path)

  private[GBMClassificationModel] class GBMClassificationModelWriter(
      instance: GBMClassificationModel)
      extends MLWriter {

    private case class Data(weight: Double, subspace: SubSpace)

    override protected def saveImpl(path: String): Unit = {
      val extraJson =
        ("numClasses" -> instance.numClasses) ~ ("numBaseModels" -> instance.numBaseModels)
      GBMClassifierParams.saveImpl(instance, path, sc, Some(extraJson))
      instance.models.zipWithIndex.foreach {
        case (models, idx) =>
          models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach {
            case (model, k) =>
              val modelPath = new Path(path, s"model-$k-$idx").toString
              model.save(modelPath)
          }
      }
      instance.weights.zip(instance.subspaces).zipWithIndex.foreach {
        case ((weights, subspace), idx) =>
          weights.zipWithIndex.foreach {
            case (weight, k) =>
              val data = Data(weight, subspace)
              val dataPath = new Path(path, s"data-$k-$idx").toString
              sparkSession.createDataFrame(Seq(data)).repartition(1).write.json(dataPath)
          }
      }

    }
  }

  private class GBMClassificationModelReader extends MLReader[GBMClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBMClassificationModel].getName

    override def load(path: String): GBMClassificationModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = GBMClassifierParams.loadImpl(path, sc, className)
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val numModels = (metadata.metadata \ "numBaseModels").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        (0 until numClasses).map { k =>
          val modelPath = new Path(path, s"model-$k-$idx").toString
          DefaultParamsReader
            .loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
        }.toArray
      }
      val boostsData = (0 until numModels).toArray.map { idx =>
        (0 until numClasses)
          .map { k =>
            val dataPath = new Path(path, s"data-$k-$idx").toString
            val data = sparkSession.read.json(dataPath).select("weight", "subspace").head()
            (data.getAs[Double](0), data.getAs[Seq[Long]](1).map(_.toInt).toArray)
          }
          .toArray
          .unzip
      }.unzip
      val bcModel =
        new GBMClassificationModel(
          metadata.uid,
          numClasses,
          boostsData._1,
          boostsData._2.map(_(0)),
          models)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
