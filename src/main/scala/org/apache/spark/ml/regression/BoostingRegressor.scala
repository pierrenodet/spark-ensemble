package org.apache.spark.ml.regression

import org.apache.commons.math3.distribution.PoissonDistribution
import org.apache.commons.math3.util.FastMath
import org.apache.spark.ml.bagging.BaggingPredictor
import org.apache.spark.ml.boosting.{BoostedPredictionModel, BoostingParams}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel

trait BoostingRegressorParams extends BoostingParams {
  setDefault(reduce -> { predictions: Array[Double] =>
    predictions.sum / predictions.length
  })
}

class BoostingRegressor(override val uid: String)
    extends Predictor[Vector, BoostingRegressor, BoostingRegressionModel]
    with BoostingRegressorParams
    with BaggingPredictor {

  def setBaseLearner(
    value: Predictor[_, _, _]
  ): this.type =
    set(baseLearner, value.asInstanceOf[PredictorVectorType])

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
    val spark = dataset.sparkSession

    val regressor = getBaseLearner
    setBaseLearner(
      regressor
        .set(regressor.labelCol, getLabelCol)
        .set(regressor.featuresCol, getFeaturesCol)
        .set(regressor.predictionCol, getPredictionCol)
    )

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, maxIter, seed)

    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    if (!isDefined(weightCol) || $(weightCol).isEmpty) setWeightCol("weight")

    val instances: RDD[Instance] =
      dataset.select(col($(labelCol)), w, col($(featuresCol))).rdd.map {
        case Row(label: Double, weight: Double, features: Vector) =>
          Instance(label, weight, features)
      }

    def trainBooster(
      baseLearner: Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]],
      learningRate: Double,
      seed: Long,
      loss: Double => Double
    )(instances: RDD[Instance]): (BoostedPredictionModel, RDD[Instance]) = {

      val labelColName = baseLearner.getLabelCol
      val featuresColName = baseLearner.getFeaturesCol

      val agg =
        instances.map { case Instance(_, weight, _) => (1, weight) }.reduce {
          case ((i1, w1), (i2, w2)) => (i1 + i2, w1 + w2)
        }
      val numLines: Int = agg._1
      val sumWeights: Double = agg._2

      val normalized = instances.map {
        case Instance(label, weight, features) =>
          Instance(label, weight / sumWeights, features)
      }
      val sampled = normalized.zipWithIndex().flatMap {
        case (Instance(label, weight, features), i) =>
          val poisson = new PoissonDistribution(weight * numLines)
          poisson.reseedRandomGenerator(seed + i)
          Iterator.fill(poisson.sample())(Instance(label, weight, features))
      }

      if (sampled.isEmpty) {
        val bpm = new BoostedPredictionModel(1, 0, null)
        return (bpm, instances)
      }

      val sampledDF = spark.createDataFrame(sampled).toDF(labelColName, "weight", featuresColName)
      val model = baseLearner.fit(sampledDF)

      val errors = instances.map {
        case Instance(label, _, features) => FastMath.abs(model.predict(features) - label)
      }
      val errorMax = errors.max()
      val losses = errors.map(error => loss(error / errorMax))
      val estimatorError =
        normalized.map(_.weight).zip(losses).map { case (weight, loss) => weight * loss }.reduce(_ + _)

      if (estimatorError <= 0) {
        val bpm = new BoostedPredictionModel(0, 1, model)
        return (bpm, instances)
      }

      val beta = estimatorError / (1 - estimatorError)
      val estimatorWeight = learningRate * FastMath.log(1 / beta)
      val instancesWithNewWeights = instances.zip(losses).map {
        case (Instance(label, weight, features), loss) => Instance(label, weight * FastMath.pow(beta, loss), features)
      }
      val bpm = new BoostedPredictionModel(estimatorError, estimatorWeight, model)
      (bpm, instancesWithNewWeights)

    }

    def trainBoosters(
      baseLearner: Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]],
      learningRate: Double,
      seed: Long,
      loss: Double => Double
    )(
      instances: RDD[Instance],
      acc: Array[BoostedPredictionModel],
      iter: Int
    ): Array[BoostedPredictionModel] = {

      val persistedInput = if (instances.getStorageLevel == StorageLevel.NONE) {
        instances.persist(StorageLevel.MEMORY_AND_DISK)
        true
      } else {
        false
      }

      val (bpm, updated) =
        trainBooster(baseLearner, learningRate, seed + iter, loss)(
          instances
        )

      if (iter == 0) {
        if (persistedInput) instances.unpersist()
        acc
      } else {
        trainBoosters(baseLearner, learningRate, seed + iter, loss)(
          updated,
          acc ++ Array(bpm),
          iter - 1
        )
      }
    }

    val models =
      trainBoosters(getBaseLearner, getLearningRate, getSeed, getLoss)(
        instances,
        Array.empty,
        getMaxIter
      )

    val usefulModels = models.filter(_.getWeight > 0)

    new BoostingRegressionModel(usefulModels)

  }

}

class BoostingRegressionModel(override val uid: String, models: Array[BoostedPredictionModel])
    extends PredictionModel[Vector, BoostingRegressionModel]
    with BoostingRegressorParams {

  def this(models: Array[BoostedPredictionModel]) = this(Identifiable.randomUID("BoostingRegressionModel"), models)

  override def predict(features: Vector): Double = getReduce(weightedPredictions(features, models))

  def weightedPredictions(features: Vector, models: Array[BoostedPredictionModel]): Array[Double] = {
    models.map(model => {
      model.getWeight * model.getModel.predict(features)
    })
  }

  override def copy(extra: ParamMap): BoostingRegressionModel = {
    val copied = new BoostingRegressionModel(uid, models)
    copyValues(copied, extra).setParent(parent)
  }
}
