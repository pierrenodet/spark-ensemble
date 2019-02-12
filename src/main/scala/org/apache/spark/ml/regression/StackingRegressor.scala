package org.apache.spark.ml.regression

import org.apache.spark.ml.bagging.BaggingPredictor
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.stacking.StackingParams
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.util.ThreadUtils

import scala.concurrent.Future
import scala.concurrent.duration.Duration

class StackingRegressor(override val uid: String)
    extends Predictor[Vector, StackingRegressor, StackingRegressionModel]
    with StackingParams
    with BaggingPredictor {

  def setLearners(
    value: Array[Predictor[_, _, _]]
  ): this.type =
    set(learners, value.map(_.asInstanceOf[PredictorVectorType]))

  def setStacker(
    value: Predictor[_, _, _]
  ): this.type =
    set(stacker, value.asInstanceOf[PredictorVectorType])

  def setParallelism(value: Int): this.type = set(parallelism, value)

  def this() = this(Identifiable.randomUID("StackingRegressor"))

  override def copy(extra: ParamMap): StackingRegressor = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): StackingRegressionModel = instrumented { instr =>
    val spark = dataset.sparkSession

    setLearners(
      getLearners.map(
        learner =>
          learner
            .set(learner.labelCol, getLabelCol)
            .set(learner.featuresCol, getFeaturesCol)
            .set(learner.predictionCol, "prediction")
      )
    )

    val tmp = getStacker
    setStacker(
      tmp
        .set(tmp.labelCol, "label")
        .set(tmp.featuresCol, "features")
        .set(tmp.predictionCol, getPredictionCol)
    )

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, seed)

    val df = dataset.toDF().cache()

    val learners = getLearners
    val numLearners = learners.length

    val futureModels = (0 until numLearners).map(
      iter =>
        Future[PredictionModel[Vector, _]] {

          instr.logDebug(s"Start training for $iter learner")

          val model = learners(iter).fit(df)

          instr.logDebug(s"Start training for $iter learner")

          model

        }(getExecutionContext)
    )

    val models = futureModels.map(f => ThreadUtils.awaitResult(f, Duration.Inf)).toArray

    val points: RDD[LabeledPoint] =
      df.select(col($(labelCol)), col($(featuresCol))).rdd.map {
        case Row(label: Double, features: Vector) =>
          LabeledPoint(label, features)
      }

    val predictions =
      points.map(point => LabeledPoint(point.label, Vectors.dense(models.map(model => model.predict(point.features)))))

    val predictionsDF = spark.createDataFrame(predictions)

    val stack = getStacker.fit(predictionsDF)

    df.unpersist()

    new StackingRegressionModel(models, stack)

  }

}

class StackingRegressionModel(
  override val uid: String,
  models: Array[PredictionModel[Vector, _]],
  stack: PredictionModel[Vector, _]
) extends PredictionModel[Vector, StackingRegressionModel]
    with BoostingRegressorParams {

  def this(models: Array[PredictionModel[Vector, _]], stack: PredictionModel[Vector, _]) =
    this(Identifiable.randomUID("StackingRegressionModel"), models, stack)

  override def predict(features: Vector): Double = {
    stack.predict(Vectors.dense(models.map(_.predict(features))))
  }

  override def copy(extra: ParamMap): StackingRegressionModel = {
    val copied = new StackingRegressionModel(uid, models, stack)
    copyValues(copied, extra).setParent(parent)
  }
}
