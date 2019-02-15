package org.apache.spark.ml.regression

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.ensemble.{
  EnsemblePredictionModelType,
  EnsemblePredictorType,
  HasBaseLearners,
  HasStacker
}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.apache.spark.ml.stacking.StackingParams
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.util.ThreadUtils
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{parse, render}
import org.json4s.{DefaultFormats, JObject}

import scala.concurrent.Future
import scala.concurrent.duration.Duration

trait StackingRegressorParams extends StackingParams {}

object StackingRegressorParams {

  def saveImpl(
      instance: StackingRegressorParams,
      path: String,
      sc: SparkContext,
      extraMetadata: Option[JObject] = None): Unit = {

    val params = instance.extractParamMap().toSeq
    val jsonParams = render(
      params
        .filter { case ParamPair(p, _) => p.name != "baseLearners" && p.name != "stacker" }
        .map { case ParamPair(p, v) => p.name -> parse(p.jsonEncode(v)) }
        .toList)

    DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, Some(jsonParams))
    HasBaseLearners.saveImpl(instance, path, sc)
    HasStacker.saveImpl(instance, path, sc)

  }

  def loadImpl(path: String, sc: SparkContext, expectedClassName: String)
    : (DefaultParamsReader.Metadata, Array[EnsemblePredictorType], EnsemblePredictorType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val learners = HasBaseLearners.loadImpl(path, sc)
    val stacker = HasStacker.loadImpl(path, sc)
    (metadata, learners, stacker)

  }

}

class StackingRegressor(override val uid: String)
    extends Predictor[Vector, StackingRegressor, StackingRegressionModel]
    with StackingRegressorParams
    with MLWritable {

  def setBaseLearners(value: Array[Predictor[_, _, _]]): this.type =
    set(baseLearners, value.map(_.asInstanceOf[EnsemblePredictorType]))

  def setStacker(value: Predictor[_, _, _]): this.type =
    set(stacker, value.asInstanceOf[EnsemblePredictorType])

  def setParallelism(value: Int): this.type = set(parallelism, value)

  def this() = this(Identifiable.randomUID("StackingRegressor"))

  override def copy(extra: ParamMap): StackingRegressor = {
    val copied = new StackingRegressor(uid)
    copyValues(copied, extra)
    copied.setBaseLearners(copied.getBaseLearners.map(_.copy(extra)))
    copied.setStacker(copied.getStacker.copy(extra))
  }
  override protected def train(dataset: Dataset[_]): StackingRegressionModel = instrumented {
    instr =>
      val spark = dataset.sparkSession

      setBaseLearners(
        getBaseLearners.map(
          learner =>
            learner
              .set(learner.labelCol, getLabelCol)
              .set(learner.featuresCol, getFeaturesCol)
              .set(learner.predictionCol, "prediction")))

      val tmp = getStacker
      setStacker(
        tmp
          .set(tmp.labelCol, "label")
          .set(tmp.featuresCol, "features")
          .set(tmp.predictionCol, getPredictionCol))

      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(this, seed)

      val df = dataset.toDF().cache()

      val learners = getBaseLearners
      val numLearners = learners.length

      val futureModels = (0 until numLearners).map(iter =>
        Future[EnsemblePredictionModelType] {

          instr.logDebug(s"Start training for $iter learner")

          val model = learners(iter).fit(df)

          instr.logDebug(s"Start training for $iter learner")

          model

        }(getExecutionContext))

      val models = futureModels.map(f => ThreadUtils.awaitResult(f, Duration.Inf)).toArray

      val points: RDD[LabeledPoint] =
        df.select(col($(labelCol)), col($(featuresCol))).rdd.map {
          case Row(label: Double, features: Vector) =>
            LabeledPoint(label, features)
        }

      val predictions =
        points.map(
          point =>
            LabeledPoint(
              point.label,
              Vectors.dense(models.map(model => model.predict(point.features)))))

      val predictionsDF = spark.createDataFrame(predictions)

      val stack = getStacker.fit(predictionsDF)

      df.unpersist()

      new StackingRegressionModel(models, stack)

  }

  override def write: MLWriter = new StackingRegressor.StackingRegressorWriter(this)

}

object StackingRegressor extends MLReadable[StackingRegressor] {

  override def read: MLReader[StackingRegressor] = new StackingRegressorReader

  override def load(path: String): StackingRegressor = super.load(path)

  private[StackingRegressor] class StackingRegressorWriter(instance: StackingRegressor)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      StackingRegressorParams.saveImpl(instance, path, sc)
    }

  }

  private class StackingRegressorReader extends MLReader[StackingRegressor] {

    /** Checked against metadata when loading model */
    private val className = classOf[StackingRegressor].getName

    override def load(path: String): StackingRegressor = {
      val (metadata, learners, stacker) = StackingRegressorParams.loadImpl(path, sc, className)
      val sr = new StackingRegressor(metadata.uid)
      metadata.getAndSetParams(sr)
      sr.setBaseLearners(learners.map(_.asInstanceOf[Predictor[Vector, _, _]]))
      sr.setStacker(stacker)
    }
  }

}

class StackingRegressionModel(
    override val uid: String,
    val models: Array[EnsemblePredictionModelType],
    val stack: EnsemblePredictionModelType)
    extends PredictionModel[Vector, StackingRegressionModel]
    with StackingRegressorParams
    with MLWritable
    with Serializable {

  def this(models: Array[EnsemblePredictionModelType], stack: EnsemblePredictionModelType) =
    this(Identifiable.randomUID("StackingRegressionModel"), models, stack)

  override def predict(features: Vector): Double = {
    stack.predict(Vectors.dense(models.map(_.predict(features))))
  }

  override def copy(extra: ParamMap): StackingRegressionModel = {
    val copied = new StackingRegressionModel(uid, models, stack)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new StackingRegressionModel.StackingRegressionModelWriter(this)

}

object StackingRegressionModel extends MLReadable[StackingRegressionModel] {

  override def read: MLReader[StackingRegressionModel] = new StackingRegressionModelReader

  override def load(path: String): StackingRegressionModel = super.load(path)

  private[StackingRegressionModel] class StackingRegressionModelWriter(
      instance: StackingRegressionModel)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      StackingRegressorParams.saveImpl(instance, path, sc)
      instance.models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach {
        case (model, idx) =>
          val modelPath = new Path(path, s"model-$idx").toString
          model.save(modelPath)
      }
      val stackPath = new Path(path, "stack").toString
      instance.stack.asInstanceOf[MLWritable].save(stackPath)

    }
  }

  private class StackingRegressionModelReader extends MLReader[StackingRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[StackingRegressionModel].getName

    override def load(path: String): StackingRegressionModel = {
      implicit val format = DefaultFormats
      val (metadata, learners, stacker) = StackingRegressorParams.loadImpl(path, sc, className)
      val models = learners.indices.toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
      }
      val stackPath = new Path(path, "stack").toString
      val stack =
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](stackPath, sc)
      val srModel = new StackingRegressionModel(metadata.uid, models, stack)
      metadata.getAndSetParams(srModel)
      srModel
    }
  }
}
