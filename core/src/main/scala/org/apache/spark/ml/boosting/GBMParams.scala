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

package org.apache.spark.ml.boosting

import java.util.UUID

import breeze.linalg.{DenseVector => BreezeDV}
import breeze.optimize.{
  ApproximateGradientFunction,
  CachedDiffFunction,
  DiffFunction,
  LBFGS => BreezeLBFGS,
  LBFGSB => BreezeLBFGSB
}
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

private[ml] trait GBMParams
    extends PredictorParams
    with HasLearningRate
    with HasMaxIter
    with BoostingParams
    with HasSubBag {

  setDefault(learningRate -> 1.0)
  setDefault(numBaseLearners -> 10)
  setDefault(tol -> 1e-3)
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
      labelColName: String,
      currentPredictionColName: String,
      boosterPredictionColName: String,
      loss: (Double, Double) => Double,
      grad: (Double, Double) => Double,
      maxIter: Int,
      tol: Double)(df: DataFrame): Double = {

    val transformed = df
      .select(col(labelColName), col(currentPredictionColName), col(boosterPredictionColName))
      .cache()

    val cdf = new CachedDiffFunction[BreezeDV[Double]](new DiffFunction[BreezeDV[Double]] {
      override def calculate(denseVector: BreezeDV[Double]): (Double, BreezeDV[Double]) = {
        val x = denseVector(0)
        val df = transformed
        val l = loss
        val ludf =
          udf[Double, Double, Double, Double](
            (label: Double, current: Double, prediction: Double) =>
              l(label, current + x * prediction))
        val g = grad
        val gudf = udf[Double, Double, Double, Double](
          (label: Double, current: Double, prediction: Double) =>
            g(label, current + x * prediction) * prediction)
        val lcn = labelColName
        val bpcn = boosterPredictionColName
        val cpcn = currentPredictionColName
        val res = df
          .agg(
            sum(ludf(col(lcn), col(cpcn), col(bpcn))),
            sum(gudf(col(lcn), col(cpcn), col(bpcn))))
          .first()
        (res.getDouble(0), BreezeDV[Double](Array(res.getDouble(1))))
      }
    })

    val lbfgsb =
      new BreezeLBFGSB(
        BreezeDV[Double](Array(Double.NegativeInfinity)),
        BreezeDV[Double](Array(Double.PositiveInfinity)),
        maxIter = maxIter,
        tolerance = tol,
        m = 10)
    val optimized = lbfgsb.minimize(cdf, BreezeDV[Double](Array(0.0)))

    optimized(0)
  }

  def findOptimizedWeight(
      labelColName: String,
      currentPredictionColName: String,
      boosterPredictionColName: String,
      loss: (Double, Double) => Double,
      maxIter: Int,
      tol: Double)(df: DataFrame): Double = {

    val transformed = df
      .select(col(labelColName), col(currentPredictionColName), col(boosterPredictionColName))
      .cache()

    def f(denseVector: BreezeDV[Double]): Double = {
      val x = denseVector(0)
      val df = transformed
      val l = loss
      val ludf =
        udf[Double, Double, Double, Double]((label: Double, current: Double, booster: Double) =>
          l(label, current + x * booster))
      val lcn = labelColName
      val bpcn = boosterPredictionColName
      val cpcn = currentPredictionColName
      val res = df
        .agg(sum(ludf(col(lcn), col(cpcn), col(bpcn))))
        .first()
      res.getDouble(0)
    }

    val agf = new ApproximateGradientFunction(f)

    val cdf = new CachedDiffFunction[BreezeDV[Double]](agf)

    val lbfgsb =
      new BreezeLBFGSB(
        BreezeDV[Double](Array(Double.NegativeInfinity)),
        BreezeDV[Double](Array(Double.PositiveInfinity)),
        maxIter = maxIter,
        tolerance = tol,
        m = 10)
    val optimized = lbfgsb.minimize(cdf, BreezeDV[Double](Array(0.0)))

    optimized(0)

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

    val cdf = new CachedDiffFunction[BreezeDV[Double]](new DiffFunction[BreezeDV[Double]] {
      override def calculate(denseVector: BreezeDV[Double]): (Double, BreezeDV[Double]) = {
        val x = denseVector(0)
        val df = transformed
        val l = loss
        val ludf =
          udf[Double, Double]((label: Double) => l(label, x))
        val g = grad
        val gudf = udf[Double, Double]((label: Double) => g(label, x))
        val lcn = labelColName
        val res = df
          .agg(sum(ludf(col(lcn))), sum(gudf(col(lcn))))
          .first()
        (res.getDouble(0), BreezeDV[Double](Array(res.getDouble(1))))
      }
    })

    val lbfgsb =
      new BreezeLBFGSB(
        BreezeDV[Double](Array(Double.NegativeInfinity)),
        BreezeDV[Double](Array(Double.PositiveInfinity)),
        maxIter = maxIter,
        tolerance = tol,
        m = 10)
    val optimized = lbfgsb.minimize(cdf, BreezeDV[Double](Array(0.0)))

    optimized(0)
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
        udf[Double, Double]((label: Double) => l(label, x))
      val lcn = labelColName
      val res = df
        .agg(sum(ludf(col(lcn))))
        .first()
      res.getDouble(0)
    }

    val agf = new ApproximateGradientFunction(f)

    val cdf = new CachedDiffFunction[BreezeDV[Double]](agf)

    val lbfgsb =
      new BreezeLBFGSB(
        BreezeDV[Double](Array(Double.NegativeInfinity)),
        BreezeDV[Double](Array(Double.PositiveInfinity)),
        maxIter = maxIter,
        tolerance = tol,
        m = 10)
    val optimized = lbfgsb.minimize(cdf, BreezeDV[Double](Array(0.0)))

    optimized(0)

  }

  def evaluateOnValidation(
      model: GBMRegressionModel,
      labelColName: String,
      loss: (Double, Double) => Double)(df: DataFrame): Double = {
    val lossUDF = udf[Double, Double, Double](loss)
    if (df.isEmpty) {
      Double.MaxValue
    } else {
      model
        .transform(df)
        .agg(sum(lossUDF(col(labelColName), col(model.getPredictionCol))))
        .head()
        .getDouble(0)
    }
  }

  def evaluateOnValidation(
      model: GBMClassificationModel,
      labelColName: String,
      loss: (Double, Double) => Double)(df: DataFrame): Double = {
    val lossUDF = udf[Double, Double, Double](loss)
    if (df.isEmpty) {
      Double.MaxValue
    } else {
      val transformed = model
        .transform(df)
      val vecToArrUDF =
        udf[Array[Double], Vector]((features: Vector) => features.toArray)
      Range(0, model.numClasses).toArray
        .map(
          k =>
            transformed
              .withColumn(
                labelColName,
                when(col(labelColName) === k.toDouble, 1.0).otherwise(0.0))
              .agg(sum(lossUDF(
                col(labelColName),
                element_at(vecToArrUDF(col(model.getRawPredictionCol)), k + 1))))
              .head()
              .getDouble(0))
        .sum
    }
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
