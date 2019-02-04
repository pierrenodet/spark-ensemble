package org.apache.spark.ml.regression

import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasMaxIter, HasParallelism}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}
import org.apache.spark.sql.Dataset
import org.apache.spark.util.ThreadUtils

import scala.concurrent.Future
import scala.concurrent.duration.Duration
import scala.util.Random

trait BaggingRegressorParams extends PredictorParams with HasMaxIter with HasParallelism {

  /**
    * param for the estimator to be stacked with bagging
    *
    * @group param
    */
  val baseLearner: Param[Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]]] = new Param(this, "baseLearner", "base learner that will get stacked with bagging")

  /** @group getParam */
  def getBaseLearner: Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]] = $(baseLearner)

  /**
    * param for whether samples are drawn with replacement
    *
    * @group param
    */
  val replacement: Param[Boolean] = new BooleanParam(this, "replacement", "whether samples are drawn with replacement")

  /** @group getParam */
  def getReplacement: Boolean = $(replacement)

  setDefault(replacement -> true)

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val sampleRatio: Param[Double] = new DoubleParam(this, "sampleRatio", "ratio of rows sampled out of the dataset")

  /** @group getParam */
  def getSampleRatio: Double = $(sampleRatio)

  setDefault(sampleRatio -> 1)

  /**
    * param for whether samples are drawn with replacement
    *
    * @group param
    */
  val replacementFeatures: Param[Boolean] = new BooleanParam(this, "replacementFeautres", "whether features sampling are drawn with replacement")

  /** @group getParam */
  def getReplacementFeatures: Boolean = $(replacementFeatures)

  setDefault(replacementFeatures -> true)

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val sampleFeatureRatio: Param[Double] = new DoubleParam(this, "sampleFeatureRatio", "ratio of features sampled out of the dataset")

  /** @group getParam */
  def getSampleFeatureRatio: Double = $(sampleFeatureRatio)

  setDefault(sampleRatio -> 1)

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val seed: Param[Long] = new LongParam(this, "seed", "seed for randomness")

  /** @group getParam */
  def getSeed: Long = $(seed)

  setDefault(seed -> System.nanoTime())

}

class BaggingRegressor(override val uid: String) extends Predictor[Vector, BaggingRegressor, BaggingRegressionModel] with BaggingRegressorParams {

  def this() = this(Identifiable.randomUID("BaggingRegressor"))

  // Parameters from BaggingRegressorParams:

  /** @group setParam */
  def setBaseLearner(value: Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]]): this.type = set(baseLearner, value)

  /** @group setParam */
  def setReplacement(value: Boolean): this.type = set(replacement, value)

  /** @group setParam */
  def setSampleRatio(value: Double): this.type = set(sampleRatio, value)

  /** @group setParam */
  def setReplacementFeatures(value: Boolean): this.type = set(replacementFeatures, value)

  /** @group setParam */
  def setSampleFeatureRatio(value: Double): this.type = set(sampleFeatureRatio, value)

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /**
    * Set the maximum level of parallelism to evaluate models in parallel.
    * Default is 1 for serial evaluation
    *
    * @group expertSetParam
    */
  def setParallelism(value: Int): this.type = set(parallelism, value)

  override def copy(extra: ParamMap): BaggingRegressor = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): BaggingRegressionModel = instrumented { instr =>

    val df = dataset.toDF()

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, maxIter, seed, parallelism)

    def sampleFeatures(withReplacement: Boolean, sampleRatio: Double)(features: Vector, seed: Long): Vector = {

      val n = (sampleRatio * features.size).toInt
      val rnd = new Random(seed)

      new DenseVector(Array.fill(n)(features(rnd.nextInt(features.size))))

    }

    //val sampleFeaturesUDF = df.sparkSession.udf.register("sampleFeatures", sampleFeatures(getReplacementFeatures, getSampleFeatureRatio))

    val futureModels = (0 to getMaxIter).map(iter =>
      Future[PredictionModel[Vector, _]] {

        val train = df.sample(getReplacement, getSampleRatio, getSeed + iter)
        val test = df.except(train)
        //val fullySampled = train.withColumn("sampledFeaturesCol", sampleFeaturesUDF(df.col(getFeaturesCol), (getSeed + iter)))

        instr.logDebug(s"Start training for $iter iteration on $train with $getBaseLearner")

        val model = getBaseLearner.fit(train)

        instr.logDebug(s"Training done for $iter iteration on $train with $getBaseLearner")

        model

      }(getExecutionContext))

    val models = futureModels.map(ThreadUtils.awaitResult(_, Duration.Inf))

    new BaggingRegressionModel(models.toArray)

  }


}

class BaggingRegressionModel(override val uid: String, models: Array[PredictionModel[Vector, _]]) extends PredictionModel[Vector, BaggingRegressionModel] with BaggingRegressorParams {

  def this(models: Array[PredictionModel[Vector, _]]) = this(Identifiable.randomUID("BaggingRegressionModel"), models)

  override def predict(features: Vector): Double = predictNormal(features)

  def predictNormal(features: Vector): Double = {
    val predictions = models.map(model =>
      model.predict(features))
    predictions.sum / predictions.length
  }

  def predictFuture(features: Vector): Double = {
    val futurePredictions = models.map(model => Future[Double] {
      model.predict(features)
    }(getExecutionContext))
    val predictions = futurePredictions.map(ThreadUtils.awaitResult(_, Duration.Inf))
    predictions.sum / predictions.length
  }

  override def copy(extra: ParamMap): BaggingRegressionModel = defaultCopy(extra)

  def getModels: Array[PredictionModel[Vector, _]] = models

}
