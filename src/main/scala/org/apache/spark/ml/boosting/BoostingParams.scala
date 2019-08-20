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

trait BoostingParams
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
  setDefault(tol -> 1E-3)

  def evaluateOnValidation(
      weights: Array[Double],
      boosters: Array[EnsemblePredictionModelType],
      labelColName: String,
      featuresColName: String,
      loss: Double => Double)(df: DataFrame): Double = {
    val model = new BoostingRegressionModel(weights, boosters)
    val lossUDF = udf(loss)
    model
      .transform(df)
      .agg(sum(lossUDF(abs(col(labelColName) - col(model.getPredictionCol)))))
      .head()
      .getDouble(0)
  }

  def evaluateOnValidation(
      numClasses: Int,
      weights: Array[Double],
      boosters: Array[EnsemblePredictionModelType],
      labelColName: String,
      featuresColName: String,
      loss: Double => Double)(df: DataFrame): Double = {
    val model = new BoostingClassificationModel(numClasses, weights, boosters)
    val lossUDF = udf(loss)
    model
      .transform(df)
      .agg(
        sum(lossUDF(when(col(labelColName) === col(model.getPredictionCol), 0.0).otherwise(1.0))))
      .head()
      .getDouble(0)
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
    math.log(1 / beta)
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
    if (avgl < ((numClasses - 1.0) / numClasses) && avgl > 0) {
      terminateVal(withValidation, error, verror, tol, numRound, numTry, iter, instrumentation)
    } else {
      instrumentation.logInfo(s"Stopped because weight of new booster is higher than 0.5")
      (0, 0.0, 1)
    }
  }

}
