package org.apache.spark.ml.regression

import org.apache.spark.ml.bagging.{BaggingParams, BaggingPredictionModel, BaggingPredictor, PatchedPredictionModel}
import org.apache.spark.ml.classification.{BaggingClassificationModel, BaggingClassifier}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.sql.Dataset
import org.apache.spark.util.ThreadUtils

import scala.concurrent.Future
import scala.concurrent.duration.Duration

trait BaggingRegressorParams extends BaggingParams {
  setDefault(reduce -> { predictions: Array[Double] =>
    predictions.sum / predictions.length
  })
}

class BaggingRegressor(override val uid: String)
    extends Predictor[Vector, BaggingRegressor, BaggingRegressionModel]
    with BaggingRegressorParams
    with BaggingPredictor {

  def this() = this(Identifiable.randomUID("BaggingRegressor"))

  // Parameters from BaggingRegressorParams:

  /** @group setParam */
  def setBaseLearner(
    value: Predictor[_, _, _]
  ): this.type =
    set(baseLearner, value.asInstanceOf[PredictorVectorType])

  /** @group setParam */
  def setReplacement(value: Boolean): this.type = set(replacement, value)

  /** @group setParam */
  def setSampleRatio(value: Double): this.type = set(sampleRatio, value)

  /** @group setParam */
  def setReplacementFeatures(value: Boolean): this.type = set(replacementFeatures, value)

  /** @group setParam */
  def setSampleRatioFeatures(value: Double): this.type = set(sampleRatioFeatures, value)

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

  override def copy(extra: ParamMap): BaggingRegressor = {
    val copied = new BaggingRegressor(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  override protected def train(dataset: Dataset[_]): BaggingRegressionModel = instrumented { instr =>
    val classifier = getBaseLearner
    setBaseLearner(
      classifier
        .set(classifier.labelCol, getLabelCol)
        .set(classifier.featuresCol, getFeaturesCol)
        .set(classifier.predictionCol, getPredictionCol)
    )

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, maxIter, seed, parallelism)

    val withBag =
      dataset.toDF().transform(withWeightedBag(getReplacement, getSampleRatio, getMaxIter, getSeed, "weightedBag"))

    val df = withBag.cache()

    val futureModels = (0 until getMaxIter).map(
      iter =>
        Future[PatchedPredictionModel] {

          val rowSampled = df.transform(withSampledRows("weightedBag", iter))

          val numFeatures = getNumFeatures(df, getFeaturesCol)
          val featuresIndices: Array[Int] =
            arrayIndicesSample(getReplacementFeatures, (getSampleRatioFeatures * numFeatures).toInt, getSeed + iter)(
              (0 until numFeatures).toArray
            )
          val rowFeatureSampled = rowSampled.transform(withSampledFeatures(getFeaturesCol, featuresIndices))

          instr.logDebug(s"Start training for $iter iteration on $rowFeatureSampled with $getBaseLearner")

          val model = getBaseLearner.fit(rowFeatureSampled)

          instr.logDebug(s"Training done for $iter iteration on $rowFeatureSampled with $getBaseLearner")

          new PatchedPredictionModel(featuresIndices, model)

        }(getExecutionContext)
    )

    val models = futureModels.map(f => ThreadUtils.awaitResult(f, Duration.Inf))

    df.unpersist()

    new BaggingRegressionModel(models.toArray)

  }
}

class BaggingRegressionModel(override val uid: String, models: Array[PatchedPredictionModel])
    extends RegressionModel[Vector, BaggingRegressionModel]
    with BaggingRegressorParams
    with BaggingPredictionModel {

  def this(models: Array[PatchedPredictionModel]) = this(Identifiable.randomUID("BaggingRegressionModel"), models)

  def this() = this(Array.empty)

  override def predict(features: Vector): Double = getReduce(predictNormal(features, models))

  override def copy(extra: ParamMap): BaggingRegressionModel = {
    val copied = new BaggingRegressionModel(uid, models)
    copyValues(copied, extra).setParent(parent)
  }
  def getModels: Array[PredictionModel[Vector, _]] = models.map(_.getModel)

}
