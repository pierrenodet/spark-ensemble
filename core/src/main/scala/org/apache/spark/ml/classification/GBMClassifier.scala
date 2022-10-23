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

import org.apache.commons.math3.optim.MaxEval
import org.apache.commons.math3.optim.MaxIter
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.univariate.BrentOptimizer
import org.apache.commons.math3.optim.univariate.SearchInterval
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.boosting.GBMParams
import org.apache.spark.ml.boosting._
import org.apache.spark.ml.ensemble.EnsemblePredictionModelType
import org.apache.spark.ml.ensemble.EnsembleRegressorType
import org.apache.spark.ml.ensemble.HasBaseLearner
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.util.PeriodicRDDCheckpointer
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import org.json4s.DefaultFormats
import org.json4s.JObject
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.JsonMethods.render

import java.util.Locale
import org.apache.spark.ml.ensemble.Utils
import org.apache.spark.ml.param.ParamValidators
import org.apache.spark.ml.classification.Classifier
import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.dummy.DummyClassifier
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.classification.ProbabilisticClassifier

private[ml] trait GBMClassifierParams extends GBMParams {

  /**
   * strategy for the init predictions, can be the class-prior or the uniform distribution.
   * (case-insensitive) Supported: "uniform", "prior". (default = prior)
   *
   * @group param
   */
  val initStrategy: Param[String] =
    new Param(
      this,
      "initStrategy",
      s"""strategy for the init predictions, can be a constant optimized for the minimized loss, zero, or the base learner learned on labels, (case-insensitive). Supported options: ${GBMClassifierParams.supportedInitStrategy
          .mkString(",")}""",
      ParamValidators.inArray(GBMClassifierParams.supportedInitStrategy))

  /** @group getParam */
  def getInitStrategy: String = $(initStrategy).toLowerCase(Locale.ROOT)

  /**
   * Loss function which Boosting tries to minimize. (case-insensitive) Supported: "log-loss",
   * "exponential". (default = log-loss)
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

  setDefault(loss -> "log-loss")

  /**
   * Influence trimming deletes all observations with the lowest norm on the second derivative of
   * the loss to improve computation times
   *
   * @group param
   */
  val influenceTrimmingRatio: Param[Double] =
    new DoubleParam(
      this,
      "influenceTrimmingRatio",
      "Influence trimming deletes all observations with the lowest norm on the second derivative of the loss to improve computation times.")

  /** @group getParam */
  def getInfluenceTrimmingRatio: Double = $(influenceTrimmingRatio)

  setDefault(influenceTrimmingRatio -> 0.0)

}

private[ml] object GBMClassifierParams {

  final val supportedLossTypes: Array[String] =
    Array("log-loss", "exponential").map(_.toLowerCase(Locale.ROOT))

  final val supportedInitStrategy: Array[String] =
    Array("uniform", "prior").map(_.toLowerCase(Locale.ROOT))

  def loss(loss: String, numClasses: Int): GBMClassificationLoss =
    loss match {
      case "log-loss" => LogLoss(numClasses)
      case "exponential" => ExponentialLoss
      case _ => throw new RuntimeException(s"GBMClassifier was given bad loss type: $loss")
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
      expectedClassName: String): (DefaultParamsReader.Metadata, EnsembleRegressorType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val learner = HasBaseLearner.loadImpl[EnsembleRegressorType](path, sc)
    (metadata, learner)
  }

}

class GBMClassifier(override val uid: String)
    extends ProbabilisticClassifier[Vector, GBMClassifier, GBMClassificationModel]
    with GBMClassifierParams
    with MLWritable {

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setBaseLearner(value: EnsembleRegressorType): this.type =
    set(baseLearner, value)

  /** @group setParam */
  def setNumBaseLearners(value: Int): this.type = set(numBaseLearners, value)

  /** @group setParam */
  def setReplacement(value: Boolean): this.type = set(replacement, value)

  /** @group setParam */
  def setSubsampleRatio(value: Double): this.type = set(subsampleRatio, value)

  /** @group setParam */
  def setSubspaceRatio(value: Double): this.type = set(subspaceRatio, value)

  /** @group setParam */
  def setInitStrategy(value: String): this.type = set(initStrategy, value)

  /** @group setParam */
  def setLoss(value: String): this.type = set(loss, value)

  /** @group setParam */
  def setInfluenceTrimmingRatio(value: Double): this.type = set(influenceTrimmingRatio, value)

  /** @group setParam */
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  /** @group expertSetParam */
  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  def setTol(value: Double): this.type = set(tol, value)

  /** @group expertSetParam */
  def setOptimizedWeights(value: Boolean): this.type = set(optimizedWeights, value)

  /** @group expertSetParam */
  def setUpdates(value: String): this.type = set(updates, value)

  /** @group setParam */
  def setValidationIndicatorCol(value: String): this.type = set(validationIndicatorCol, value)

  /** @group setParam */
  def setValidationTol(value: Double): this.type = set(validationTol, value)

  /** @group setParam */
  def setNumRounds(value: Int): this.type = set(numRounds, value)

  /** @group setParam */
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  def this() = this(Identifiable.randomUID("GBMClassifier2"))

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
        initStrategy,
        loss,
        influenceTrimmingRatio,
        numBaseLearners,
        learningRate,
        optimizedWeights,
        validationIndicatorCol,
        subsampleRatio,
        replacement,
        subspaceRatio,
        maxIter,
        tol,
        seed)

      val spark = dataset.sparkSession
      val sc = spark.sparkContext

      val withValidation = isDefined(validationIndicatorCol) && $(validationIndicatorCol).nonEmpty

      val (trainDataset, validationDataset) = if (withValidation) {
        (
          extractInstances(dataset.filter(not(col($(validationIndicatorCol))))),
          extractInstances(dataset.filter(col($(validationIndicatorCol)))))
      } else {
        (extractInstances(dataset), null)
      }

      trainDataset
        .persist(StorageLevel.MEMORY_AND_DISK)
      if (withValidation) validationDataset.persist(StorageLevel.MEMORY_AND_DISK)

      val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
      val numClasses = getNumClasses(dataset)
      instr.logNumClasses(numClasses)
      validateNumClasses(numClasses)

      val models = Array.ofDim[EnsemblePredictionModelType]($(numBaseLearners))
      val subspaces =
        Array.tabulate($(numBaseLearners))(i =>
          subspace($(subspaceRatio), numFeatures, $(seed) + i))
      val weights = Array.ofDim[Double]($(numBaseLearners))

      val init = new DummyClassifier()
        .setStrategy(getInitStrategy)
        .fit(spark.createDataFrame(trainDataset))

      val gbmLoss = GBMClassifierParams.loss(getLoss, numClasses)
      //   val gbmDiffFunction = new GBMDiffFunction(gbmLoss, trainDataset, $(aggregationDepth))

      //   val optimizer = new BrentOptimizer($(tol), $(tol))

      //   var predictions = trainDataset.map(instance => init.predict(instance.features))
      //   val predictionsCheckpointer = new PeriodicRDDCheckpointer[Double](
      //     $(checkpointInterval),
      //     sc,
      //     StorageLevel.MEMORY_AND_DISK)
      //   predictionsCheckpointer.update(predictions)

      //   var validationPredictions: RDD[Double] = null
      //   var validatePredictionsCheckpointer: PeriodicRDDCheckpointer[Double] = null
      //   var bestValidationError: Double = 0.0
      //   if (withValidation) {
      //     validationPredictions = validationDataset.map(instance => init.predict(instance.features))
      //     validatePredictionsCheckpointer = new PeriodicRDDCheckpointer[Double](
      //       $(checkpointInterval),
      //       sc,
      //       StorageLevel.MEMORY_AND_DISK)
      //     validatePredictionsCheckpointer.update(validationPredictions)
      //     bestValidationError = validationDataset
      //       .zip(validationPredictions)
      //       .map { case (instance, prediction) =>
      //         gbmLoss.loss(instance.label, prediction)
      //       }
      //       .mean()
      //   }

      //   var i = 0
      //   var v = 0
      //   while (i < $(numBaseLearners) && v < $(numRounds)) {

      //     val subspace = subspaces(i)

      //     val bag = trainDataset
      //       .zip(predictions)
      //       .sample($(replacement), $(subsampleRatio), $(seed))

      //     val subbag =
      //       bag.map { case (instance, prediction) =>
      //         (instance.copy(features = slice(subspace)(instance.features)), prediction)
      //       }

      //     subbag.persist(StorageLevel.MEMORY_AND_DISK)

      //     val pseudoResiduals = gbmLoss match {
      //       case gbmLoss: HasScalarHessian if (getUpdates == "newton") =>
      //         val hessians = subbag.map { case (instance, prediction) =>
      //           gbmLoss.hessian(instance.label, prediction)
      //         }
      //         val sumHessians = hessians.treeReduce(_ + _, $(aggregationDepth))
      //         subbag.zip(hessians).map { case ((instance, prediction), hessian) =>
      //           val negGrad = gbmLoss.negativeGradient(instance.label, prediction)
      //           (instance
      //             .copy(
      //               label = negGrad / hessian,
      //               weight = 1.0 / 2.0 * hessian / sumHessians * instance.weight))
      //         }
      //       case gbmLoss =>
      //         subbag.map { case (instance, prediction) =>
      //           instance.copy(label = gbmLoss.negativeGradient(instance.label, prediction))
      //         }
      //     }

      //     val featuresMetadata =
      //       Utils.getFeaturesMetadata(dataset, $(featuresCol), Some(subspace))

      //     val df = spark
      //       .createDataFrame(pseudoResiduals)
      //       .withMetadata("features", featuresMetadata)

      //     val model =
      //       fitBaseLearner($(baseLearner), "label", "features", $(predictionCol), Some("weight"))(
      //         df)

      //     val solution = $(optimizedWeights) match {
      //       case true =>
      //         val instances = subbag.map(_._1)
      //         val predictions = subbag.map(_._2)
      //         val directions = subbag
      //           .map { case (instance, _) => model.predict(instance.features) }
      //           .persist(StorageLevel.MEMORY_AND_DISK)
      //         val gbmDiffFunction = new GBMDiffFunction(gbmLoss, instances, $(aggregationDepth))
      //         val gbmLineSearch =
      //           GBMLineSearch.functionFromSearchDirection(gbmDiffFunction, predictions, directions)
      //         val objective = new UnivariateObjectiveFunction(x => gbmLineSearch.valueAt(x));
      //         val searchInterval = new SearchInterval(0, 10, 1);
      //         val alpha = optimizer
      //           .optimize(
      //             objective,
      //             searchInterval,
      //             GoalType.MINIMIZE,
      //             new MaxIter($(maxIter)),
      //             new MaxEval($(maxIter)))
      //           .getPoint()
      //         directions.unpersist()
      //         alpha
      //       case false => 1.0
      //     }

      //     val weight = $(learningRate) * solution

      //     models(i) = model
      //     weights(i) = weight

      //     subbag.unpersist()

      //     val directions = trainDataset.map(instance => model.predict(instance.features))

      //     predictions = predictions
      //       .zip(directions)
      //       .map { case (prediction, direction) =>
      //         prediction + weight * direction
      //       }
      //     predictionsCheckpointer.update(predictions)

      //     if (withValidation) {
      //       validationPredictions = validationDataset
      //         .zip(validationPredictions)
      //         .map { case (instance, prediction) =>
      //           prediction + weight * model.predict(instance.features)
      //         }
      //       validatePredictionsCheckpointer.update(validationPredictions)
      //       val validationError = validationDataset
      //         .zip(validationPredictions)
      //         .map { case (instance, prediction) =>
      //           gbmLoss.loss(instance.label, prediction)
      //         }
      //         .mean()
      //       if (bestValidationError - validationError < $(validationTol) * math.max(
      //           validationError,
      //           0.01)) {
      //         v += 1
      //       } else if (validationError < bestValidationError) {
      //         bestValidationError = validationError
      //         v = 0
      //       }
      //     }

      //     i += 1

      //   }

      trainDataset.unpersist()
      if (withValidation) validationDataset.unpersist()

      new GBMClassificationModel(
        numClasses,
        weights.take(1),
        subspaces.take(1),
        models.take(1),
        init)

    }

  override def write: MLWriter =
    new GBMClassifier.GBMClassifier2Writer(this)

}

object GBMClassifier extends MLReadable[GBMClassifier] {

  override def read: MLReader[GBMClassifier] = new GBMClassifierReader

  override def load(path: String): GBMClassifier = super.load(path)

  private[GBMClassifier] class GBMClassifier2Writer(instance: GBMClassifier) extends MLWriter {

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

class GBMClassificationModel(
    override val uid: String,
    override val numClasses: Int,
    val weights: Array[Double],
    val subspaces: Array[Array[Int]],
    val models: Array[EnsemblePredictionModelType],
    val init: EnsemblePredictionModelType)
    extends ProbabilisticClassificationModel[Vector, GBMClassificationModel]
    with GBMClassifierParams
    with MLWritable {

  def this(
      numClasses: Int,
      weights: Array[Double],
      subspaces: Array[Array[Int]],
      models: Array[EnsemblePredictionModelType],
      init: EnsemblePredictionModelType) =
    this(
      Identifiable.randomUID("BoostingRegressionModel"),
      numClasses,
      weights,
      subspaces,
      models,
      init)

  val numModels: Int = models.length

  private val gbmLoss = GBMClassifierParams.loss(getLoss, numClasses)

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector =
    gbmLoss.raw2probabilityInPlace(rawPrediction)

  override def predictRaw(features: Vector): Vector = ???

  override def copy(extra: ParamMap): GBMClassificationModel = {
    val copied = new GBMClassificationModel(uid, numClasses, weights, subspaces, models, init)
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

    private case class Data(weight: Double, subspace: Array[Int])

    override protected def saveImpl(path: String): Unit = {
      GBMClassifierParams.saveImpl(
        instance,
        path,
        sc,
        Some(("numModels" -> instance.numModels) ~ ("numClasses" -> instance.numClasses)))
      val initPath = new Path(path, s"init").toString
      instance.init.asInstanceOf[MLWritable].save(initPath)
      instance.models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach { case (model, idx) =>
        val modelPath = new Path(path, s"model-$idx").toString
        model.save(modelPath)
      }
      instance.weights.zip(instance.subspaces).zipWithIndex.foreach {
        case ((weight, subspace), idx) =>
          val data = Data(weight, subspace)
          val dataPath = new Path(path, s"data-$idx").toString
          sparkSession.createDataFrame(Seq(data)).repartition(1).write.json(dataPath)
      }
    }
  }

  private class GBMClassificationModelReader extends MLReader[GBMClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBMClassificationModel].getName

    override def load(path: String): GBMClassificationModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = GBMClassifierParams.loadImpl(path, sc, className)
      val numModels = (metadata.metadata \ "numModels").extract[Int]
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val initPath = new Path(path, s"init").toString
      val init = DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](initPath, sc)
      val models = (0 until numModels).toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
      }
      val boostsData = (0 until numModels).toArray.map { idx =>
        val dataPath = new Path(path, s"data-$idx").toString
        val data = sparkSession.read.json(dataPath).select("weight", "subspace").head()
        (data.getAs[Double](0), data.getAs[Seq[Long]](1).map(_.toInt).toArray)
      }.unzip
      val gbmcModel =
        new GBMClassificationModel(
          metadata.uid,
          numClasses,
          boostsData._1,
          boostsData._2,
          models,
          init)
      metadata.getAndSetParams(gbmcModel)
      gbmcModel
    }
  }
}
