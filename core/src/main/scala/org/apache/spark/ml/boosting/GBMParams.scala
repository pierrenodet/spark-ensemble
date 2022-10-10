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

import breeze.linalg.{DenseVector => BreezeDV}
import breeze.optimize.ApproximateGradientFunction
import breeze.optimize.CachedDiffFunction
import breeze.optimize.DiffFunction
import breeze.optimize.{LBFGS => BreezeLBFGS}
import breeze.optimize.{LBFGSB => BreezeLBFGSB}
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.classification.GBMClassificationModel
import org.apache.spark.ml.ensemble._
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.ml.linalg.BLAS
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.regression.GBMRegressionModel
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.sql.Column
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import java.util.UUID
import scala.util.Random
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Encoders
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.sql.types.StructType
import scala.reflect.ClassTag
import org.apache.spark.sql.catalyst.encoders.RowEncoder

private[ml] trait GBMParams
    extends PredictorParams
    with HasLearningRate
    with HasMaxIter
    with BoostingParams
    with HasSubBag {

  setDefault(learningRate -> 0.1)
  setDefault(numBaseLearners -> 10)
  setDefault(tol -> 1e-6)
  setDefault(maxIter -> 100)

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

  protected def findOptimizedWeight(
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
        (res.getDouble(0), BreezeDV[Double](res.getDouble(1)))
      }
    })

    val lbfgsb =
      new BreezeLBFGSB(
        BreezeDV[Double](Double.NegativeInfinity),
        BreezeDV[Double](Double.PositiveInfinity),
        maxIter = maxIter,
        tolerance = tol,
        m = 100)
    val optimized = lbfgsb.minimize(cdf, BreezeDV[Double](0.0))
    optimized(0)
  }

  protected def findOptimizedWeight(
      labelColName: String,
      currentPredictionColName: String,
      boosterPredictionColName: String,
      loss: (Vector, Vector) => Double,
      grad: (Vector, Vector) => Vector,
      numClasses: Int,
      maxIter: Int,
      tol: Double)(df: DataFrame): Array[Double] = {

    val transformed = df
      .select(col(labelColName), col(currentPredictionColName), col(boosterPredictionColName))
      .cache()

    val cdf = new CachedDiffFunction[BreezeDV[Double]](new DiffFunction[BreezeDV[Double]] {
      override def calculate(weights: BreezeDV[Double]): (Double, BreezeDV[Double]) = {
        def xpyz(x: Vector, y: Vector, z: Vector): Vector = {
          val dim = x.size
          val res = Array.ofDim[Double](dim)
          var i = 0
          while (i < dim) {
            res(i) = x(i) + y(i) * z(i)
            i += 1
          }
          Vectors.dense(res)
        }
        val df = transformed
        val bcWeights = df.sparkSession.sparkContext.broadcast(Vectors.fromBreeze(weights))
        val l = loss
        val ludf =
          udf[Double, Vector, Vector, Vector](
            (label: Vector, current: Vector, prediction: Vector) => {
              l(label, xpyz(current, bcWeights.value, prediction))
            })
        val g = grad
        val gudf = udf[Vector, Vector, Vector, Vector](
          (label: Vector, current: Vector, prediction: Vector) => {
            xpyz(
              Vectors.zeros(label.size),
              g(label, xpyz(current, bcWeights.value, prediction)),
              prediction)
          })
        val lcn = labelColName
        val bpcn = boosterPredictionColName
        val cpcn = currentPredictionColName
        var agg = Seq.empty[Column]
        var k = 0
        val atUDF = udf[Double, Vector, Int]((vector: Vector, k: Int) => vector(k))
        while (k < numClasses) {
          agg = agg :+ sum(atUDF(gudf(col(lcn), col(cpcn), col(bpcn)), lit(k)))
          k += 1
        }
        val res = df.agg(sum(ludf(col(lcn), col(cpcn), col(bpcn))), agg: _*).first()
        (
          res.getDouble(0),
          BreezeDV[Double](Array.range(0, numClasses).map(k => res.getDouble(k + 1))))
      }
    })

    val lbfgs = new BreezeLBFGSB(
      BreezeDV.fill(numClasses)(Double.NegativeInfinity),
      BreezeDV.fill(numClasses)(Double.PositiveInfinity),
      maxIter = maxIter,
      tolerance = tol,
      m = 10)
    val optimized = lbfgs.minimize(cdf, BreezeDV.zeros(numClasses))

    transformed.unpersist()

    optimized.toArray

  }

  protected def findOptimizedConst(
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
        (res.getDouble(0), BreezeDV[Double](res.getDouble(1)))
      }
    })

    val lbfgsb =
      new BreezeLBFGSB(
        BreezeDV[Double](Double.NegativeInfinity),
        BreezeDV[Double](Double.PositiveInfinity),
        maxIter = maxIter,
        tolerance = tol,
        m = 10)
    val optimized = lbfgsb.minimize(cdf, BreezeDV[Double](0.0))

    optimized(0)
  }

  protected def evaluateOnValidation(
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

  protected def evaluateOnValidation(
      model: GBMClassificationModel,
      labelColName: String,
      loss: (Vector, Vector) => Double)(df: DataFrame): Double = {
    if (df.isEmpty) {
      Double.MaxValue
    } else {
      val transformed = model
        .transform(df)
      val lossUDF = udf[Double, Vector, Vector](loss)
      model
        .transform(df)
        .agg(sum(lossUDF(col(labelColName), col(model.getRawPredictionCol))))
        .head()
        .getDouble(0)
    }
  }

  protected def terminate(
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

  protected def terminate(
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
