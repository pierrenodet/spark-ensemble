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
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.boosting.GBMParams
import org.apache.spark.ml.ensemble._
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.ml.linalg.BLAS
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.Matrices
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.ml.param.ParamValidators
import org.apache.spark.ml.param.shared.HasParallelism
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.ThreadUtils
import org.json4s.DefaultFormats
import org.json4s.JObject
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.JsonMethods.render

import java.util.Locale
import java.util.UUID
import scala.annotation.tailrec
import scala.concurrent.ExecutionContext
import scala.concurrent.Future
import scala.concurrent.duration.Duration

private[ml] trait GBMClassifierParams
    extends ProbabilisticClassifierParams
    with GBMParams
    with HasParallelism {

  /**
   * Loss function which Boosting tries to minimize. (case-insensitive) Supported: "divergence".
   * (default = divergence)
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

  val instanceTrimmingRatio: Param[Double] = new DoubleParam(
    this,
    "instanceTrimmingRatio",
    "instance trimming of top quantile highest residuals every step",
    ParamValidators.inRange(0, 1))

  def getInstanceTrimmingRatio: Double = $(instanceTrimmingRatio)

  setDefault(instanceTrimmingRatio -> 1.0)

  protected def trim(negGradColName: String, instanceTrimmingRatio: Double, tol: Double)(
      df: DataFrame): DataFrame = {
    val instanceWeightColName = "gbm$instance-weight" + UUID.randomUUID().toString
    val instanced = df
      .withColumn(
        instanceWeightColName,
        aggregate(
          transform(vector_to_array(col(negGradColName)), r => abs(r) * (lit(1.0) - abs(r))),
          lit(1.0),
          (acc, w) => acc * w))

    val bottom =
      instanced.stat.approxQuantile(instanceWeightColName, Array(1 - instanceTrimmingRatio), tol)(
        0)
    val trimmed =
      instanced.filter(col(instanceWeightColName) >= bottom).drop(instanceWeightColName)

    return trimmed
  }

}

private[ml] object GBMClassifierParams {

  final val supportedLossTypes: Array[String] =
    Array("divergence").map(_.toLowerCase(Locale.ROOT))

  def lossFunction(loss: String): (Vector, Vector) => Double = loss match {
    case "divergence" =>
      (y, score) => {
        val dim = score.size
        var i = 0
        var res = 0.0
        var sum = 0.0
        while (i < dim) {
          sum += math.exp(score(i))
          i += 1
        }
        i = 0
        while (i < dim) {
          res += -y(i) * (score(i) - math.log(sum))
          i += 1
        }
        res
      }
    case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")
  }

  def gradFunction(loss: String): (Vector, Vector) => Vector = loss match {
    case "divergence" =>
      (y, score) => {
        val dim = score.size
        var i = 0
        var res = Array.ofDim[Double](dim)
        var sum = 0.0
        while (i < dim) {
          sum += math.exp(score(i))
          i += 1
        }
        i = 0
        while (i < dim) {
          res(i) += -(y(i) - math.exp(score(i)) / sum)
          i += 1
        }
        Vectors.dense(res)
      }
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
    extends ProbabilisticClassifier[Vector, GBMClassifier, GBMClassificationModel]
    with GBMClassifierParams
    with MLWritable {

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setBaseLearner(value: Predictor[_, _, _]): this.type =
    set(baseLearner, value.asInstanceOf[EnsemblePredictorType])

  /** @group setParam */
  def setNumBaseLearners(value: Int): this.type = set(numBaseLearners, value)

  /** @group setParam */
  def setParallelism(value: Int): this.type = set(parallelism, value)

  /** @group setParam */
  def setLoss(value: String): this.type = set(loss, value)

  /** @group setParam */
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  /** @group setParam */
  def setOptimizedWeights(value: Boolean): this.type = set(optimizedWeights, value)

  /** @group setParam */
  def setInstanceTrimmingRatio(value: Double): this.type = set(instanceTrimmingRatio, value)

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
        instanceTrimmingRatio,
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

      val df = if (weightColIsUsed) {
        dataset.select($(labelCol), $(featuresCol), $(weightCol))
      } else {
        dataset.select($(labelCol), $(featuresCol))
      }

      val withValidation = isDefined(validationIndicatorCol) && $(validationIndicatorCol).nonEmpty

      val (train, validation) = if (withValidation) {
        (
          df.filter(not(col($(validationIndicatorCol)))),
          df.filter(col($(validationIndicatorCol))))
      } else {
        (df, df.sparkSession.emptyDataFrame)
      }

      val handlePersistence =
        dataset.storageLevel == StorageLevel.NONE && (train.storageLevel == StorageLevel.NONE) && (validation.storageLevel == StorageLevel.NONE)
      if (handlePersistence) {
        train.persist(StorageLevel.MEMORY_AND_DISK)
        validation.persist(StorageLevel.MEMORY_AND_DISK)
      }

      val numFeatures = MetadataUtils.getNumFeatures(train, getFeaturesCol)

      val numClasses = getNumClasses(train, maxNumClasses = numFeatures)
      instr.logNumClasses(numClasses)
      validateNumClasses(numClasses)

      dataset
        .select(col($(labelCol)))
        .rdd
        .map { case Row(label: Double) => label }
        .foreach(validateLabel(_, numClasses))

      val oneHotCol = "gbm$oneHot" + UUID.randomUUID().toString
      val ohem = new OneHotEncoder()
        .setInputCol($(labelCol))
        .setOutputCol(oneHotCol)
        .setDropLast(false)
        .fit(dataset)
      val onehotted = ohem.transform(dataset)
      val onehottedValidation = if (validation.isEmpty) validation else ohem.transform(validation)

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
          withValidation: Boolean,
          baseLearner: EnsemblePredictorType,
          numBaseLearners: Int,
          learningRate: Double,
          loss: (Vector, Vector) => Double,
          grad: (Vector, Vector) => Vector,
          sampleFeatureRatio: Double,
          numFeatures: Int,
          optimizedWeights: Boolean,
          instanceTrimmingRatio: Double,
          maxIter: Int,
          tol: Double,
          numRound: Int,
          seed: Long,
          instrumentation: Instrumentation)(
          weights: Array[Array[Double]],
          subspaces: Array[Array[Int]],
          boosters: Array[Array[EnsemblePredictionModelType]],
          consts: Array[Double],
          iter: Int,
          error: Double,
          numTry: Int): (
          Array[Array[Double]],
          Array[Array[Int]],
          Array[Array[EnsemblePredictionModelType]]) = {

        if (iter == 0) {

          instrumentation.logInfo(s"Learning of GBM finished.")
          (weights.dropRight(numTry), subspaces.dropRight(numTry), boosters.dropRight(numTry))

        } else {

          instrumentation.logNamedValue("iteration", numBaseLearners - iter)

          val currentPredictionColName = "gbm$current" + UUID.randomUUID().toString
          val currentRawPredictionColName = "gbm$current-raw" + UUID.randomUUID().toString
          val currentProbabilityColName = "gbm$current-proba" + UUID.randomUUID().toString
          val current =
            new GBMClassificationModel(numClasses, weights, subspaces, boosters, consts)
              .setRawPredictionCol(currentRawPredictionColName)
              .setPredictionCol(currentPredictionColName)
              .setProbabilityCol(currentProbabilityColName)
              .setFeaturesCol(featuresColName)

          val subspace = mkSubspace(sampleFeatureRatio, numFeatures, seed)

          val negGradColName = "gbm$neg-grad" + UUID.randomUUID().toString
          val negGradUDF =
            udf[Vector, Vector, Vector]((label: Vector, score: Vector) => {
              val res = grad(label, score)
              BLAS.scal(-1.0, res)
              res
            })

          val negGrad = current
            .transform(train)
            .withColumn(
              negGradColName,
              negGradUDF(col(labelColName), col(currentRawPredictionColName)))

          val trimmed = trim(negGradColName, instanceTrimmingRatio, tol)(negGrad)

          val boosterFutures = Array
            .range(0, numClasses)
            .map(k =>
              Future {

                val residualsColName = "gbm$residuals" + UUID.randomUUID().toString
                val residuals = trimmed
                  .withColumn(
                    residualsColName,
                    element_at(vector_to_array(col(negGradColName)), k + 1))

                val (subspace, subbagged) = subbag(
                  getFeaturesCol,
                  getReplacement,
                  getSubsampleRatio,
                  getSubspaceRatio,
                  numFeatures,
                  getSeed + iter)(residuals)

                val booster = fitBaseLearner(
                  baseLearner,
                  residualsColName,
                  featuresColName,
                  predictionColName,
                  weightColName)(subbagged)

                booster

              }(executionContext))

          val booster = boosterFutures
            .map(ThreadUtils.awaitResult(_, Duration.Inf))

          val kbooster = new GBMClassificationModel(
            numClasses,
            Array(Array.fill(numClasses)(1.0)),
            Array(subspace),
            Array(booster),
            consts)

          val optimizedWeight = if (getOptimizedWeights) {

            val kboosterRawPredictionColName = "gbm$kbooster-raw" + UUID
              .randomUUID()
              .toString
            val transformed =
              kbooster
                .setRawPredictionCol(kboosterRawPredictionColName)
                .transform(negGrad)

            findOptimizedWeight(
              labelColName,
              currentRawPredictionColName,
              kboosterRawPredictionColName,
              loss,
              grad,
              numClasses,
              maxIter,
              tol)(transformed)

          } else {
            Array.fill(numClasses)(1.0)
          }

          val weight = optimizedWeight.map(_ * learningRate)

          instrumentation.logNamedValue("weight", weight)

          val updatedWeights = weights :+ weight
          val updatedBoosters = boosters :+ booster
          val updatedSubspaces = subspaces :+ subspace

          val updatedModel = new GBMClassificationModel(
            numClasses,
            updatedWeights,
            updatedSubspaces,
            updatedBoosters,
            consts)

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
            withValidation,
            baseLearner,
            numBaseLearners,
            learningRate,
            loss,
            grad,
            sampleFeatureRatio,
            numFeatures,
            optimizedWeights,
            instanceTrimmingRatio,
            maxIter,
            tol,
            numRound,
            seed + iter,
            instrumentation)(
            updatedWeights,
            updatedSubspaces,
            updatedBoosters.asInstanceOf[Array[Array[EnsemblePredictionModelType]]],
            consts,
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

      val consts = Array.fill(numClasses)(0.0)

      val (weights, subspaces, boosters) =
        trainBoosters(
          onehotted,
          onehottedValidation,
          oneHotCol,
          optWeightColName,
          getFeaturesCol,
          getPredictionCol,
          getRawPredictionCol,
          numClasses,
          executionContext,
          withValidation,
          getBaseLearner,
          getNumBaseLearners,
          getLearningRate,
          GBMClassifierParams.lossFunction(getLoss),
          GBMClassifierParams.gradFunction(getLoss),
          getSubspaceRatio,
          numFeatures,
          getOptimizedWeights,
          getInstanceTrimmingRatio,
          getMaxIter,
          getTol,
          getNumRound,
          getSeed,
          instr)(
          Array.empty,
          Array.empty,
          Array.empty,
          consts,
          getNumBaseLearners,
          Double.MaxValue,
          0)

      if (handlePersistence) {
        df.unpersist()
      }

      new GBMClassificationModel(numClasses, weights, subspaces, boosters, consts)

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
    val subspaces: Array[Array[Int]],
    val models: Array[Array[EnsemblePredictionModelType]],
    val consts: Array[Double])
    extends ProbabilisticClassificationModel[Vector, GBMClassificationModel]
    with GBMClassifierParams
    with MLWritable {

  def this(
      numClasses: Int,
      weights: Array[Array[Double]],
      subspaces: Array[Array[Int]],
      models: Array[Array[EnsemblePredictionModelType]],
      consts: Array[Double]) =
    this(
      Identifiable.randomUID("GBMClassificationModel"),
      numClasses,
      weights,
      subspaces,
      models,
      consts)

  val numBaseModels: Int = models.length

  override def predictRaw(features: Vector): Vector = {

    val res = Array.ofDim[Double](numClasses)

    var i = 0
    while (i < numBaseModels) {
      var j = 0
      while (j < numClasses) {
        res(j) += models(i)(j).predict(slicer(subspaces(i))(features)) * weights(i)(j) + consts(j)
        j += 1
      }
      i += 1
    }

    return Vectors.dense(res)

  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        val values = dv.values

        // get the maximum margin
        val maxMarginIndex = rawPrediction.argmax
        val maxMargin = rawPrediction(maxMarginIndex)

        if (maxMargin == Double.PositiveInfinity) {
          var k = 0
          while (k < numClasses) {
            values(k) = if (k == maxMarginIndex) 1.0 else 0.0
            k += 1
          }
        } else {
          var sum = 0.0
          var k = 0
          while (k < numClasses) {
            values(k) = if (maxMargin > 0) {
              math.exp(values(k) - maxMargin)
            } else {
              math.exp(values(k))
            }
            sum += values(k)
            k += 1
          }
          BLAS.scal(1 / sum, dv)
        }
        dv
      case sv: SparseVector =>
        throw new RuntimeException(
          "Unexpected error in GBMClassificationModel:" +
            " raw2probabilitiesInPlace encountered SparseVector")
    }
  }

  override def copy(extra: ParamMap): GBMClassificationModel = {
    val copied = new GBMClassificationModel(uid, numClasses, weights, subspaces, models, consts)
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

    private case class Data(weight: Double, subspace: Array[Int], const: Double)

    override protected def saveImpl(path: String): Unit = {
      val extraJson =
        ("numClasses" -> instance.numClasses) ~ ("numBaseModels" -> instance.numBaseModels)
      GBMClassifierParams.saveImpl(instance, path, sc, Some(extraJson))
      instance.models.zipWithIndex.foreach { case (models, idx) =>
        models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach { case (model, k) =>
          val modelPath = new Path(path, s"model-$k-$idx").toString
          model.save(modelPath)
        }
      }
      instance.weights.zip(instance.subspaces).zipWithIndex.foreach {
        case ((weights, subspace), idx) =>
          weights.zipWithIndex.foreach { case (weight, k) =>
            val data = Data(weight, subspace, instance.consts(k))
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
            val data =
              sparkSession.read.json(dataPath).select("weight", "subspace", "const").head()
            (
              data.getAs[Double](0),
              data.getAs[Seq[Long]](1).map(_.toInt).toArray,
              data.getAs[Double](2))
          }
          .toArray
          .unzip3
      }.unzip3
      val bcModel =
        new GBMClassificationModel(
          metadata.uid,
          numClasses,
          boostsData._1,
          boostsData._2.map(_(0)),
          models,
          boostsData._3(0))
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
