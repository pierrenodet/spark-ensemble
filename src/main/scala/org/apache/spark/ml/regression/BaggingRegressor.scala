package org.apache.spark.ml.regression

import org.apache.commons.math3.distribution.PoissonDistribution
import org.apache.spark.ml.bagging.BaggingParams
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.sql.{Column, Dataset, functions}
import org.apache.spark.util.ThreadUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, DoubleType, IntegerType}

import scala.concurrent.Future
import scala.concurrent.duration.Duration
import scala.util.Random

class BaggingRegressor(override val uid: String) extends Predictor[Vector, BaggingRegressor, BaggingRegressionModel] with BaggingParams {

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
  def setReduce(value: Array[Double] => Double): this.type = set(reduce, value)

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

    import org.apache.spark.ml.linalg.DenseVector
    val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
    val toArrUdf = udf(toArr)

    def sampleFeatures(withReplacement: Boolean, sampleRatio: Double, numberSamples: Int)(col: Column, seed: Long): Column = {

      val numberFeatures = size(toArrUdf(col))
      val sample = if (sampleRatio == 1) {
        array_repeat(lit(1), numberFeatures)
      } else {
        if (withReplacement) {
          require(sampleRatio > 0)
          val poisson = new PoissonDistribution(sampleRatio)
          poisson.reseedRandomGenerator(seed)
          array_repeat(lit(if (poisson.sample() > 0) 1 else 0), numberFeatures)
        } else {
          require(sampleRatio <= 1 && sampleRatio > 0)
          val rnd = new Random(seed)
          array_repeat(lit(if (rnd.nextDouble() < sampleRatio) 1 else 0), numberFeatures)
        }
      }
      array_repeat(sample, numberSamples)

    }

    def sample(withReplacement: Boolean, sampleRatio: Double, numberSamples: Int)(col: Column, seed: Long): Column = {

      if (sampleRatio == 1) {
        array_repeat(lit(1), numberSamples)
      } else {
        if (withReplacement) {
          require(sampleRatio > 0)
          val poisson = new PoissonDistribution(sampleRatio)
          poisson.reseedRandomGenerator(seed)
          array_repeat(lit(poisson.sample()), numberSamples)
        } else {
          require(sampleRatio <= 1 && sampleRatio > 0)
          val rnd = new Random(seed)
          array_repeat(lit(if (rnd.nextDouble() < sampleRatio) 1 else 0), numberSamples)
        }
      }

    }

    df.withColumn("weightedBag", sample(getReplacement, getSampleRatio, getMaxIter)(col(getFeaturesCol), getSeed))
      .withColumn("featuresBag", sampleFeatures(getReplacement, getSampleRatio, getMaxIter)(col(getFeaturesCol), getSeed))
      .show(false)


    /*def sampleFeatures(withReplacement: Boolean, sampleRatio: Double)(features: Vector, seed: Long): Vector = {

      val n = (sampleRatio * features.size).toInt
      val rnd = new Random(seed)

      new DenseVector(Array.fill(n)(features(rnd.nextInt(features.size))))

    }

    val sampleFeaturesUDF = df.sparkSession.udf.register("sampleFeatures", sampleFeatures(getReplacementFeatures, getSampleFeatureRatio))
*/
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

class BaggingRegressionModel(override val uid: String, models: Array[PredictionModel[Vector, _]]) extends PredictionModel[Vector, BaggingRegressionModel] with BaggingParams {

  def this(models: Array[PredictionModel[Vector, _]]) = this(Identifiable.randomUID("BaggingRegressionModel"), models)

  override def predict(features: Vector): Double = getReduce(predictNormal(features))

  def predictNormal(features: Vector): Array[Double] = {
    models.map(model =>
      model.predict(features))
  }

  def predictFuture(features: Vector): Array[Double] = {
    val futurePredictions = models.map(model => Future[Double] {
      model.predict(features)
    }(getExecutionContext))
    futurePredictions.map(ThreadUtils.awaitResult(_, Duration.Inf))
  }

  override def copy(extra: ParamMap): BaggingRegressionModel = defaultCopy(extra)

  def getModels: Array[PredictionModel[Vector, _]] = models

}
