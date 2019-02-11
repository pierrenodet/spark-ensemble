package org.apache.spark.ml.regression

import org.apache.spark.ml.bagging.BaggingPredictor
import org.apache.spark.ml.boosting.{BoostedPredictionModel, BoostingParams}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.bfunctions.poisson
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}
/*
class BoostingRegressor(override val uid: String)
    extends Predictor[Vector, BoostingRegressor, BoostingRegressionModel]
    with BoostingParams
    with BaggingPredictor {

  def setBaseLearner(
    value: Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]]
  ): this.type =
    set(baseLearner, value)

  /** @group setParam */
  def setReduce(value: Array[Double] => Double): this.type = set(reduce, value)

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  def this() = this(Identifiable.randomUID("BoostingRegressor"))

  override def copy(extra: ParamMap): BoostingRegressor = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): BoostingRegressionModel = instrumented { instr =>
    setBaseLearner(
      getBaseLearner
        .setFeaturesCol(getFeaturesCol)
        .asInstanceOf[Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]]]
    )
    setBaseLearner(
      getBaseLearner
        .setLabelCol(getLabelCol)
        .asInstanceOf[Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]]]
    )
    setBaseLearner(
      getBaseLearner
        .setPredictionCol(getPredictionCol)
        .asInstanceOf[Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]]]
    )

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, maxIter, seed)

    val df = if (!isDefined(weightCol) || $(weightCol).isEmpty) {
      setWeightCol("weights")
      dataset.withColumn("weights", lit(1))
    } else dataset

    def trainBooster(
      baseLearner: Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]],
      learningRate: Double,
      weightColName: String,
      labelColName: String,
      predictionColName: String,
      seed: Long,
      error: (Double, Double) => Double
    )(dataset: Dataset[_]): (BoostedPredictionModel, Dataset[_]) = {
      val agg = dataset.select(sum(weightColName).cast(DoubleType), count(col(weightColName)).cast(IntegerType)).first()
      val sumWeights: Double = agg.getDouble(0)
      val numLines: Double = agg.getInt(1)
      println(sumWeights + " " + numLines)
      val normalized = dataset.withColumn(weightColName, col(weightColName) / lit(sumWeights) * lit(numLines))
      normalized.show(false)
      dataset.printSchema()
      val sampled = normalized
        .withColumn("instanceNumber", poisson(col(weightColName), lit(seed)))
        .transform(withSampledRows("instanceNumber"))
      if (sampled.isEmpty) {
        val bpm = new BoostedPredictionModel(1, 0, null)
        return (bpm, dataset)
      }
      sampled.show(false)
      val model = baseLearner.fit(sampled)
      val predict = model.transform(dataset)
      val errorUDF = udf(error)
      val estimatorError = predict
        .select(mean(errorUDF(col(labelColName), col(predictionColName)) * col(weightColName)))
        .first()
        .getDouble(0)
      println(estimatorError)
      if (estimatorError <= 0) {
        val bpm = new BoostedPredictionModel(0, 1, model)
        return (bpm, predict.drop(getPredictionCol))
      }
      val beta = estimatorError / (1 - estimatorError)
      println(beta)
      val estimatorWeight = learningRate * math.log(1 / beta)
      println(estimatorWeight)
      val sampleWeighted = predict
        .withColumn(
          weightColName,
          pow(lit(beta), (lit(1) - abs(col(labelColName) - col(predictionColName))) * lit(learningRate))
        )
        .drop(getPredictionCol)
      sampleWeighted.show()
      val bpm = new BoostedPredictionModel(estimatorError, estimatorWeight, model)
      (bpm, sampleWeighted)

    }

    def trainBoosters(
      baseLearner: Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]],
      learningRate: Double,
      weightColName: String,
      labelColName: String,
      predictionColName: String,
      seed: Long,
      error: (Double, Double) => Double
    )(
      dataset: Dataset[_],
      acc: Array[BoostedPredictionModel],
      iter: Int
    ): Array[BoostedPredictionModel] = {
      val (bpm, df) =
        trainBooster(baseLearner, learningRate, weightColName, labelColName, predictionColName, seed + iter, error)(
          dataset
        )
      if (iter == 0) {
        acc
      } else {
        trainBoosters(baseLearner, learningRate, weightColName, labelColName, predictionColName, seed + iter, error)(
          df,
          acc ++ Array(bpm),
          iter - 1
        )
      }
    }

    val models =
      trainBoosters(getBaseLearner, getLearningRate, getWeightCol, getLabelCol, getPredictionCol, getSeed, getError)(
        df,
        Array.empty,
        getMaxIter
      )

    val usefulModels = models.filter(_.getWeight > 0)

    new BoostingRegressionModel(usefulModels)

  }

}

class BoostingRegressionModel(override val uid: String, models: Array[BoostedPredictionModel])
    extends PredictionModel[Vector, BoostingRegressionModel]
    with BoostingParams {

  def this(models: Array[BoostedPredictionModel]) = this(Identifiable.randomUID("BoostingRegressionModel"), models)

  override def predict(features: Vector): Double = getReduce(weightedPredictions(features, models))

  def weightedPredictions(features: Vector, models: Array[BoostedPredictionModel]): Array[Double] = {
    models.map(model => {
      model.getWeight * model.getModel.predict(features)
    })
  }

  override def copy(extra: ParamMap): BoostingRegressionModel = new BoostingRegressionModel(models)
}
*/