package org.apache.spark.ml.classification

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.bagging.{BaggingParams, BaggingPredictionModel, BaggingPredictor, PatchedPredictionModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.sql.Dataset
import org.apache.spark.util.ThreadUtils
import org.json4s.DefaultFormats

import scala.concurrent.Future
import scala.concurrent.duration.Duration

trait BaggingClassifierParams extends BaggingParams with ClassifierParams {}

class BaggingClassifier(override val uid: String)
    extends Predictor[Vector, BaggingClassifier, BaggingClassificationModel]
    with BaggingClassifierParams
    with BaggingPredictor
    with MLWritable {

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
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /**
    * Set the maximum level of parallelism to evaluate models in parallel.
    * Default is 1 for serial evaluation
    *
    * @group expertSetParam
    */
  def setParallelism(value: Int): this.type = set(parallelism, value)

  override def copy(extra: ParamMap): BaggingClassifier = {
    val copied = new BaggingClassifier(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  override protected def train(dataset: Dataset[_]): BaggingClassificationModel = instrumented { instr =>
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

    new BaggingClassificationModel(models.toArray)

  }

  override def write: MLWriter = new BaggingClassifier.BaggingClassifierWriter(this)

}

object BaggingClassifier extends MLReadable[BaggingClassifier] {

  override def read: MLReader[BaggingClassifier] = new BaggingClassifierReader

  override def load(path: String): BaggingClassifier = super.load(path)

  private[BaggingClassifier] class BaggingClassifierWriter(instance: BaggingClassifier) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      BaggingParams.saveImpl(path, instance, sc)
    }

  }

  private class BaggingClassifierReader extends MLReader[BaggingClassifier] {

    /** Checked against metadata when loading model */
    private val className = classOf[BaggingClassifier].getName

    override def load(path: String): BaggingClassifier = {
      val (metadata, learner) = BaggingParams.loadImpl(path, sc, className)
      val bc = new BaggingClassifier(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

class BaggingClassificationModel(override val uid: String, val models: Array[PatchedPredictionModel])
    extends PredictionModel[Vector, BaggingClassificationModel]
    with BaggingClassifierParams
    with BaggingPredictionModel
    with MLWritable {

  def this(models: Array[PatchedPredictionModel]) = this(Identifiable.randomUID("BaggingRegressionModel"), models)

  def this() = this(Array.empty)

  override def predict(features: Vector): Double = Vectors.dense(predictNormal(features,models)).argmax

  override def copy(extra: ParamMap): BaggingClassificationModel = {
    val copied = new BaggingClassificationModel(uid, models)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new BaggingClassificationModel.BaggingClassificationModelWriter(this)

}

object BaggingClassificationModel extends MLReadable[BaggingClassificationModel] {

  override def read: MLReader[BaggingClassificationModel] = new BaggingClassificationModelReader

  override def load(path: String): BaggingClassificationModel = super.load(path)

  private[BaggingClassificationModel] class BaggingClassificationModelWriter(instance: BaggingClassificationModel)
      extends MLWriter {

    private case class Data(indices: Array[Int])

    override protected def saveImpl(path: String): Unit = {
      BaggingParams.saveImpl(path, instance, sc)
      instance.models.map(_.model.asInstanceOf[MLWritable]).zipWithIndex.foreach {
        case (model, idx) =>
          val modelPath = new Path(path, s"model-$idx").toString
          model.save(modelPath)
      }
      instance.models.zipWithIndex.foreach {
        case (model, idx) =>
          val data = Data(model.indices)
          val dataPath = new Path(path, s"data-$idx").toString
          sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
      }

    }
  }

  private class BaggingClassificationModelReader extends MLReader[BaggingClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BaggingClassificationModel].getName

    override def load(path: String): BaggingClassificationModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = BaggingParams.loadImpl(path, sc, className)
      val numModels = metadata.getParamValue("maxIter").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[PredictionModel[Vector, _]](modelPath, sc)
      }
      val indices = (0 until numModels).toArray.map { idx =>
        val dataPath = new Path(path, s"data-$idx").toString
        val data = sparkSession.read.parquet(dataPath).select("indices").head()
        data.getAs[Seq[Int]](0).toArray
      }
      val bcModel = new BaggingClassificationModel(metadata.uid, indices.zip(models).map {
        case (a, b) => new PatchedPredictionModel(a, b)
      })
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
