package org.apache.spark.ml.boosting

import breeze.linalg.{DenseVector => BreezeDV}
import breeze.optimize.{ApproximateGradientFunction, CachedDiffFunction, LBFGSB => BreezeLBFGSB}
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.classification.GBMClassificationModel
import org.apache.spark.ml.ensemble.HasSubBag.SubSpace
import org.apache.spark.ml.ensemble._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.regression.GBMRegressionModel
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

trait GBMParams
    extends PredictorParams
    with HasLearningRate
    with HasMaxIter
    with BoostingParams
    with HasSubBag {

  setDefault(learningRate -> 1)
  setDefault(numBaseLearners -> 10)
  setDefault(tol -> 1E-3)
  setDefault(maxIter -> 10)

  /**
   * param for using optimized weights in GBM
   *
   * @group param
   */
  val optimizedWeights: Param[Boolean] =
    new BooleanParam(
      this,
      "optimizedWeights",
      "whether optimized weights for gbm are used or weights are fixed to 1")

  /** @group getParam */
  def getOptimizedWeights: Boolean = $(optimizedWeights)

  setDefault(optimizedWeights -> false)

  def findOptimizedWeight(
      booster: EnsemblePredictionModelType,
      labelColName: String,
      predictionColName: String,
      loss: (Double, Double) => Double,
      grad: (Double, Double) => Double,
      maxIter: Int,
      tol: Double)(df: DataFrame): Double = {

    val transformed = booster
      .transform(df)
      .select(col(labelColName), col(predictionColName))
      .cache()

    val cdf = new CachedDiffFunction[BreezeDV[Double]]({ denseVector: BreezeDV[Double] =>
      {
        val x = denseVector(0)
        val df = transformed
        val l = loss
        val ludf =
          udf[Double, Double, Double]((label: Double, prediction: Double) =>
            l(label, x * prediction))
        val g = grad
        val gudf = udf[Double, Double, Double]((label: Double, prediction: Double) =>
          g(label, x * prediction) * prediction)
        val lcn = labelColName
        val pcn = predictionColName
        val res = df
          .agg(sum(ludf(col(lcn), col(pcn))), sum(gudf(col(lcn), col(pcn))))
          .first()
        (res.getDouble(0), BreezeDV[Double](Array(res.getDouble(1))))
      }
    })

    val lbfgsb =
      new BreezeLBFGSB(
        BreezeDV[Double](Array(0.0)),
        BreezeDV[Double](Array(1.0)),
        maxIter = maxIter,
        tolerance = tol,
        m = 10)
    val iterations = lbfgsb.iterations(cdf, BreezeDV[Double](Array(1.0))).toArray

    val tmp = iterations(iterations.map(_.value).zipWithIndex.min._2).x(0)

    val optimized = if (tmp.isNaN) 0.0 else tmp

    transformed.unpersist()

    optimized
  }

  def findOptimizedWeight(
      booster: EnsemblePredictionModelType,
      labelColName: String,
      predictionColName: String,
      loss: (Double, Double) => Double,
      maxIter: Int,
      tol: Double)(df: DataFrame): Double = {

    val transformed = booster
      .transform(df)
      .select(col(labelColName), col(predictionColName))
      .cache()

    def f(denseVector: BreezeDV[Double]): Double = {
      val x = denseVector(0)
      val df = transformed
      val l = loss
      val ludf =
        udf[Double, Double, Double]((label: Double, prediction: Double) =>
          l(label, x * prediction))
      val lcn = labelColName
      val pcn = predictionColName
      val res = df
        .agg(sum(ludf(col(lcn), col(pcn))))
        .first()
      res.getDouble(0)
    }

    val agf = new ApproximateGradientFunction(f)

    val cdf = new CachedDiffFunction[BreezeDV[Double]](agf)

    val lbfgsb =
      new BreezeLBFGSB(
        BreezeDV[Double](Array(0.0)),
        BreezeDV[Double](Array(Double.PositiveInfinity)),
        maxIter = maxIter,
        tolerance = tol,
        m = 10)
    val iterations = lbfgsb.iterations(cdf, BreezeDV[Double](Array(1.0))).toArray

    val tmp = iterations(iterations.map(_.value).zipWithIndex.min._2).x(0)

    val optimized = if (tmp.isNaN) 0.0 else tmp

    transformed.unpersist()

    optimized
  }

  def findOptimizedConst(
      labelColName: String,
      loss: (Double, Double) => Double,
      grad: (Double, Double) => Double,
      maxIter: Int,
      tol: Double)(df: DataFrame): Double = {

    val transformed = df
      .select(col(labelColName))
      .cache()

    val cdf = new CachedDiffFunction[BreezeDV[Double]]({ denseVector: BreezeDV[Double] =>
      {
        val x = denseVector(0)
        val df = transformed
        val l = loss
        val ludf =
          udf[Double, Double, Double]((label: Double, prediction: Double) =>
            l(label, x * prediction))
        val g = grad
        val gudf = udf[Double, Double, Double]((label: Double, prediction: Double) =>
          g(label, x * prediction) * prediction)
        val lcn = labelColName
        val res = df
          .agg(sum(ludf(col(lcn), lit(1))), sum(gudf(col(lcn), lit(1))))
          .first()
        (res.getDouble(0), BreezeDV[Double](Array(res.getDouble(1))))
      }
    })

    val lbfgsb =
      new BreezeLBFGSB(
        BreezeDV[Double](Array(0.0)),
        BreezeDV[Double](Array(Double.PositiveInfinity)),
        maxIter = maxIter,
        tolerance = tol,
        m = 10)
    val iterations = lbfgsb.iterations(cdf, BreezeDV[Double](Array(1.0))).toArray

    val tmp = iterations(iterations.map(_.value).zipWithIndex.min._2).x(0)

    val optimized = if (tmp.isNaN) 0.0 else tmp

    transformed.unpersist()

    optimized
  }

  def findOptimizedConst(
      labelColName: String,
      loss: (Double, Double) => Double,
      maxIter: Int,
      tol: Double)(df: DataFrame): Double = {

    val transformed = df
      .select(col(labelColName))
      .cache()

    def f(denseVector: BreezeDV[Double]): Double = {
      val x = denseVector(0)
      val df = transformed
      val l = loss
      val ludf =
        udf[Double, Double, Double]((label: Double, prediction: Double) =>
          l(label, x * prediction))
      val lcn = labelColName
      val res = df
        .agg(sum(ludf(col(lcn), lit(1))))
        .first()
      res.getDouble(0)
    }

    val agf = new ApproximateGradientFunction(f)

    val cdf = new CachedDiffFunction[BreezeDV[Double]](agf)

    val lbfgsb =
      new BreezeLBFGSB(
        BreezeDV[Double](Array(0.0)),
        BreezeDV[Double](Array(Double.PositiveInfinity)),
        maxIter = maxIter,
        tolerance = tol,
        m = 10)
    val iterations = lbfgsb.iterations(cdf, BreezeDV[Double](Array(1.0))).toArray

    val tmp = iterations(iterations.map(_.value).zipWithIndex.min._2).x(0)

    val optimized = if (tmp.isNaN) 0.0 else tmp

    transformed.unpersist()

    optimized
  }

  def evaluateOnValidation(
      weights: Array[Double],
      subspaces: Array[SubSpace],
      boosters: Array[EnsemblePredictionModelType],
      const: Double,
      labelColName: String,
      featuresColName: String,
      loss: (Double, Double) => Double)(df: DataFrame): Double = {
    val model = new GBMRegressionModel(weights, subspaces, boosters, const)
    val lossUDF = udf[Double, Double, Double](loss)
    model
      .transform(df)
      .agg(sum(lossUDF(col(labelColName), col(model.getPredictionCol))))
      .head()
      .getDouble(0)
  }

  def evaluateOnValidation(
      numClasses: Int,
      weights: Array[Array[Double]],
      subspaces: Array[SubSpace],
      boosters: Array[Array[EnsemblePredictionModelType]],
      labelColName: String,
      featuresColName: String,
      loss: (Double, Double) => Double)(df: DataFrame): Double = {
    val model = new GBMClassificationModel(numClasses, weights, subspaces, boosters)
    val lossUDF = udf[Double, Double, Double](loss)
    val transformed = model
      .transform(df)
    val vecToArrUDF =
      udf[Array[Double], Vector]((features: Vector) => features.toArray)
    Range(0, numClasses).toArray
      .map(
        k =>
          transformed
            .withColumn(labelColName, when(col(labelColName) === k.toDouble, 1.0).otherwise(0.0))
            .agg(sum(lossUDF(
              col(labelColName),
              element_at(vecToArrUDF(col(model.getRawPredictionCol)), k + 1))))
            .head()
            .getDouble(0))
      .sum
  }

  def terminate(
      weights: Array[Double],
      learningRate: Double,
      withValidation: Boolean,
      error: Double,
      verror: Double,
      tol: Double,
      numRound: Int,
      numTry: Int,
      iter: Int,
      instrumentation: Instrumentation): (Int, Double, Int) = {
    if (weights.forall(_ < tol * learningRate)) {
      instrumentation.logInfo(
        s"Stopped because weight of new booster is lower than ${tol * learningRate}")
      (0, 0.0, 1)
    } else {
      terminateVal(withValidation, error, verror, tol, numRound, numTry, iter, instrumentation)
    }
  }

  def terminate(
      weight: Double,
      learningRate: Double,
      withValidation: Boolean,
      error: Double,
      verror: Double,
      tol: Double,
      numRound: Int,
      numTry: Int,
      iter: Int,
      instrumentation: Instrumentation): (Int, Double, Int) = {
    if (weight < tol * learningRate) {
      instrumentation.logInfo(
        s"Stopped because weight of new booster is lower than ${tol * learningRate}")
      (0, 0.0, 1)
    } else {
      terminateVal(withValidation, error, verror, tol, numRound, numTry, iter, instrumentation)
    }
  }

}
