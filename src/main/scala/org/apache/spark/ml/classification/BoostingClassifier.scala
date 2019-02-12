package org.apache.spark.ml.classification
import breeze.linalg.DenseVector
import org.apache.commons.math3.distribution.PoissonDistribution
import org.apache.commons.math3.util.FastMath
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.bagging.BaggingPredictor
import org.apache.spark.ml.boosting.{BoostedPredictionModel, BoostingParams}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel

trait BoostingClassifierParams extends BoostingParams with ClassifierParams {

  setDefault(reduce -> { predictions: Array[Double] =>
    val grouped = predictions.groupBy(x => x).mapValues(_.length).toSeq
    val max = grouped.map(_._2).max
    grouped.filter(_._2 == max).map(_._1).head
  }) /*
  setDefault(reduce -> { predictions: Array[Double] =>
    predictions.sum / predictions.length
  })*/
}

class BoostingClassifier(override val uid: String)
    extends Classifier[Vector, BoostingClassifier, BoostingClassificationModel]
    with BoostingClassifierParams
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

  override def copy(extra: ParamMap): BoostingClassifier = {
    val copied = new BoostingClassifier(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  override protected def train(dataset: Dataset[_]): BoostingClassificationModel = instrumented { instr =>
    val spark = dataset.sparkSession

    val classifier = getBaseLearner
    setBaseLearner(
      classifier
        .set(classifier.labelCol, getLabelCol)
        .set(classifier.featuresCol, getFeaturesCol)
        .set(classifier.predictionCol, getPredictionCol)
    )

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, maxIter, seed)

    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    if (!isDefined(weightCol) || $(weightCol).isEmpty) setWeightCol("weight")

    val numClasses = getNumClasses(dataset)

    val instances: RDD[Instance] =
      dataset.select(col($(labelCol)), w, col($(featuresCol))).rdd.map {
        case Row(label: Double, weight: Double, features: Vector) =>
          Instance(label, weight, features)
      }

    def trainBooster(
      baseLearner: PredictorVectorType,
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

      //TODO: Implement multiclass loss function
      val errors = instances.map {
        case Instance(label, _, features) => if (model.predict(features) != label) 1 else 0
      }
      val estimatorError =
        normalized.map(_.weight).zip(errors).map { case (weight, error) => weight * error }.reduce(_ + _)

      if (estimatorError <= 0) {
        val bpm = new BoostedPredictionModel(0, 1, model)
        return (bpm, instances)
      }

      val beta = estimatorError / (1 - estimatorError)
      val estimatorWeight = learningRate * (FastMath.log(1 / beta) + FastMath.log(numClasses - 1))
      val instancesWithNewWeights = instances.zip(errors).map {
        case (Instance(label, weight, features), error) =>
          Instance(label, weight * FastMath.exp(estimatorWeight * error), features)
      }
      val bpm = new BoostedPredictionModel(estimatorError, estimatorWeight, model)
      (bpm, instancesWithNewWeights)

    }

    def trainBoosters(
      baseLearner: PredictorVectorType,
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

    new BoostingClassificationModel(numClasses, usefulModels)

  }
}

class BoostingClassificationModel(
  override val uid: String,
  override val numClasses: Int,
  models: Array[BoostedPredictionModel]
) extends ClassificationModel[Vector, BoostingClassificationModel]
    with BoostingClassifierParams {

  def this(numClasses: Int, models: Array[BoostedPredictionModel]) =
    this(Identifiable.randomUID("BoostingRegressionModel"), numClasses, models)

  override protected def predictRaw(features: Vector): Vector =
    Vectors.fromBreeze(
      models
        .map(model => {
          val tmp = DenseVector.zeros[Double](numClasses)
          tmp(model.getModel.predict(features).ceil.toInt) = 1.0
          val res = model.getWeight * tmp
          res
        })
        .reduce(_ + _)
    )

  override def copy(extra: ParamMap): BoostingClassificationModel = {
    val copied = new BoostingClassificationModel(uid, numClasses, models)
    copyValues(copied, extra).setParent(parent)
  }

}
