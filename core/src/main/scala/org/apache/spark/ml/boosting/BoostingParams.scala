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
import org.apache.commons.math3.distribution.PoissonDistribution
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.classification.BoostingClassificationModel
import org.apache.spark.ml.ensemble.{
  EnsemblePredictionModelType,
  HasBaseLearner,
  HasNumBaseLearners,
  HasNumRound
}
import org.apache.spark.ml.param.shared.{HasSeed, HasTol, HasValidationIndicatorCol, HasWeightCol}
import org.apache.spark.ml.regression.BoostingRegressionModel
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import scala.util.Try

private[ml] trait BoostingParams
    extends PredictorParams
    with HasNumBaseLearners
    with HasWeightCol
    with HasSeed
    with HasBaseLearner
    with HasValidationIndicatorCol
    with HasTol
    with HasNumRound {

  setDefault(numRound -> 5)
  setDefault(numBaseLearners -> 10)
  setDefault(tol -> 1e-3)

  def evaluateOnValidation(
      weights: Array[Double],
      boosters: Array[EnsemblePredictionModelType],
      labelColName: String,
      featuresColName: String,
      loss: Double => Double)(df: DataFrame): Double = {
    val model = new BoostingRegressionModel(weights, boosters).setFeaturesCol(featuresColName)
    val lossUDF = udf(loss)
    if (df.isEmpty) {
      Double.MaxValue
    } else {
      model
        .transform(df)
        .agg(sum(lossUDF(abs(col(labelColName) - col(model.getPredictionCol)))))
        .head()
        .getDouble(0)
    }
  }

  def evaluateOnValidation(
      numClasses: Int,
      weights: Array[Double],
      boosters: Array[EnsemblePredictionModelType],
      labelColName: String,
      featuresColName: String,
      loss: Double => Double)(df: DataFrame): Double = {
    val model = new BoostingRegressionModel(weights, boosters).setFeaturesCol(featuresColName)
    val lossUDF = udf(loss)
    if (df.isEmpty) {
      Double.MaxValue
    } else {
      model
        .transform(df)
        .agg(sum(
          lossUDF(when(col(labelColName) === col(model.getPredictionCol), 0.0).otherwise(1.0))))
        .head()
        .getDouble(0)
    }
  }

  def probabilize(
      boostWeightColName: String,
      boostProbaColName: String,
      poissonProbaColName: String)(df: DataFrame): DataFrame = {
    val agg = df.agg(count(lit(1)), sum(boostWeightColName)).first()
    val (numLines, sumWeights) = (agg.getLong(0).toDouble, agg.getDouble(1))

    df.withColumn(boostProbaColName, col(boostWeightColName) / sumWeights)
      .withColumn(poissonProbaColName, col(boostProbaColName) * numLines)
  }

  def updateWeights(
      boostWeightColName: String,
      lossColName: String,
      beta: Double,
      updatedBoostWeightColName: String)(df: DataFrame): DataFrame = {
    df.withColumn(
      updatedBoostWeightColName,
      col(boostWeightColName) * pow(lit(beta), lit(1) - col(lossColName)))
  }

  def avgLoss(lossColName: String, boostProbaColName: String)(df: DataFrame): Double = {
    df.agg(sum(col(lossColName) * col(boostProbaColName)))
      .first()
      .getDouble(0)
  }

  def beta(avgl: Double, numClasses: Int = 2): Double = {
    avgl / ((1 - avgl) * (numClasses - 1))
  }

  def weight(beta: Double): Double = {
    if (beta == 0.0) {
      1.0
    } else {
      math.log(1 / beta)
    }
  }

  def extractBoostedBag(poissonProbaColName: String, seed: Long)(df: DataFrame): DataFrame = {

    val poissonProbaColIndex = df.schema.fieldIndex(poissonProbaColName)

    val replicatedRDD = df.rdd.mapPartitionsWithIndex {
      case (i, rows) =>
        rows.zipWithIndex.flatMap {
          case (row, j) =>
            val prob = row.getDouble(poissonProbaColIndex)
            if (prob == 0.0) {
              Iterator.empty
            } else {
              val poisson = new PoissonDistribution(prob)
              poisson.reseedRandomGenerator(seed + i + j)
              Iterator.fill(poisson.sample())(row)
            }
        }
    }

    df.sparkSession.createDataFrame(replicatedRDD, df.schema)

  }

  def terminateVal(
      withValidation: Boolean,
      error: Double,
      verror: Double,
      tol: Double,
      numRound: Int,
      numTry: Int,
      iter: Int,
      instrumentation: Instrumentation): (Int, Double, Int) = {
    if (withValidation) {

      val improved = verror < error * (1 - tol)
      if (improved) {
        (iter - 1, verror, 0)
      } else {
        if (numTry == numRound - 1) {
          instrumentation.logInfo(
            s"Stopped because new boosters don't improved validation performance more than $tol in $numRound rounds.")
          (0, 0.0, numTry + 1)
        } else {
          (iter - 1, error, numTry + 1)
        }
      }
    } else {
      (iter - 1, 0.0, 0)
    }

  }

  def terminate(
      avgl: Double,
      withValidation: Boolean,
      error: Double,
      verror: Double,
      tol: Double,
      numRound: Int,
      numTry: Int,
      iter: Int,
      instrumentation: Instrumentation,
      numClasses: Double = 2.0): (Int, Double, Int) = {
    if (avgl > ((numClasses - 1.0) / numClasses)) {
      instrumentation.logInfo(s"Stopped because weight of new booster is higher than $avgl")
      (0, 0.0, 1)
    } else if (avgl == 0.0) {
      instrumentation.logInfo(s"Stopped because the average loss was $avgl")
      (0, 0.0, 0)
    } else {
      terminateVal(withValidation, error, verror, tol, numRound, numTry, iter, instrumentation)
    }
  }

}
