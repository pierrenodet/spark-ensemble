package org.apache.spark.ml.classification

import org.apache.commons.math3.stat.StatUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.bagging.{Bagging, BaggingParams}
import org.apache.spark.ml.ensemble.{EnsemblePredictionModelType, EnsemblePredictorType, HasBaseLearner}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.sql.Dataset
import org.apache.spark.util.ThreadUtils
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, JObject}

import scala.concurrent.Future
import scala.concurrent.duration.Duration

trait BaggingClassifierParams extends BaggingParams with ClassifierParams {}

object BaggingClassifierParams {

  def saveImpl(
      instance: BaggingClassifierParams,
      path: String,
      sc: SparkContext,
      extraMetadata: Option[JObject] = None): Unit = {

    val params = instance.extractParamMap().toSeq
    val jsonParams = render(
      params
        .filter { case ParamPair(p, _) => p.name != "baseLearner" }
        .map { case ParamPair(p, v) => p.name -> parse(p.jsonEncode(v)) }
        .toList)

    DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, Some(jsonParams))
    HasBaseLearner.saveImpl(instance, path, sc)

  }

  def loadImpl(
      path: String,
      sc: SparkContext,
      expectedClassName: String): (DefaultParamsReader.Metadata, EnsemblePredictorType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val learner = HasBaseLearner.loadImpl(path, sc)
    (metadata, learner)

  }

}

class BaggingClassifier(override val uid: String)
    extends Predictor[Vector, BaggingClassifier, BaggingClassificationModel]
    with BaggingClassifierParams
    with MLWritable {

  def this() = this(Identifiable.randomUID("BaggingRegressor"))

  // Parameters from BaggingRegressorParams:

  /** @group setParam */
  def setBaseLearner(value: Predictor[_, _, _]): this.type =
    set(baseLearner, value.asInstanceOf[EnsemblePredictorType])

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

  override protected def train(dataset: Dataset[_]): BaggingClassificationModel = instrumented {
    instr =>
      val classifier = getBaseLearner
      setBaseLearner(
        classifier
          .set(classifier.labelCol, getLabelCol)
          .set(classifier.featuresCol, getFeaturesCol)
          .set(classifier.predictionCol, getPredictionCol))

      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(this, maxIter, seed, parallelism)

      val withBag =
        dataset
          .toDF()
          .transform(
            Bagging.withWeightedBag(getReplacement, getSampleRatio, getMaxIter, getSeed, "weightedBag"))

      val df = withBag.cache()

      val futureModels = (0 until getMaxIter).map(iter =>
        Future[(Array[Int], EnsemblePredictionModelType)] {

          val rowSampled = df.transform(Bagging.withSampledRows("weightedBag", iter))

          val numFeatures = Bagging.getNumFeatures(df, getFeaturesCol)
          val featuresIndices: Array[Int] =
            Bagging.arrayIndicesSample(
              getReplacementFeatures,
              (getSampleRatioFeatures * numFeatures).toInt,
              getSeed + iter)((0 until numFeatures).toArray)
          val rowFeatureSampled =
            rowSampled.transform(Bagging.withSampledFeatures(getFeaturesCol, featuresIndices))

          instr.logDebug(
            s"Start training for $iter iteration on $rowFeatureSampled with $getBaseLearner")

          val model = getBaseLearner.fit(rowFeatureSampled)

          instr.logDebug(
            s"Training done for $iter iteration on $rowFeatureSampled with $getBaseLearner")

          (featuresIndices, model)

        }(getExecutionContext))

      val models = futureModels.map(f => ThreadUtils.awaitResult(f, Duration.Inf))

      df.unpersist()

      new BaggingClassificationModel(models.toArray)

  }

  override def write: MLWriter = new BaggingClassifier.BaggingClassifierWriter(this)

}

object BaggingClassifier extends MLReadable[BaggingClassifier] {

  override def read: MLReader[BaggingClassifier] = new BaggingClassifierReader

  override def load(path: String): BaggingClassifier = super.load(path)

  private[BaggingClassifier] class BaggingClassifierWriter(instance: BaggingClassifier)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      BaggingClassifierParams.saveImpl(instance, path, sc)
    }

  }

  private class BaggingClassifierReader extends MLReader[BaggingClassifier] {

    /** Checked against metadata when loading model */
    private val className = classOf[BaggingClassifier].getName

    override def load(path: String): BaggingClassifier = {
      val (metadata, learner) = BaggingClassifierParams.loadImpl(path, sc, className)
      val bc = new BaggingClassifier(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

class BaggingClassificationModel(
    override val uid: String,
    val subSpaces: Array[Array[Int]],
    val models: Array[EnsemblePredictionModelType])
    extends PredictionModel[Vector, BaggingClassificationModel]
    with BaggingClassifierParams
    with MLWritable {

  def this(subSpaces: Array[Array[Int]], models: Array[EnsemblePredictionModelType]) =
    this(Identifiable.randomUID("BaggingRegressionModel"), subSpaces, models)

  def this(tuples: Array[(Array[Int], EnsemblePredictionModelType)]) =
    this(tuples.map(_._1), tuples.map(_._2))

  override def predict(features: Vector): Double =
    StatUtils.mode(subSpaces.zip(models).map {
      case (subSpace, model) =>
        val subFeatures = features match {
          case features: DenseVector => Vectors.dense(subSpace.map(features.apply))
          case features: SparseVector => features.slice(subSpace)
        }
        model.predict(subFeatures)
    }).head

  override def copy(extra: ParamMap): BaggingClassificationModel = {
    val copied = new BaggingClassificationModel(uid, subSpaces, models)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter =
    new BaggingClassificationModel.BaggingClassificationModelWriter(this)

}

object BaggingClassificationModel extends MLReadable[BaggingClassificationModel] {

  override def read: MLReader[BaggingClassificationModel] = new BaggingClassificationModelReader

  override def load(path: String): BaggingClassificationModel = super.load(path)

  private[BaggingClassificationModel] class BaggingClassificationModelWriter(
      instance: BaggingClassificationModel)
      extends MLWriter {

    private case class Data(subSpaces: Array[Int])

    override protected def saveImpl(path: String): Unit = {
      BaggingClassifierParams.saveImpl(instance, path, sc)
      instance.models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach {
        case (model, idx) =>
          val modelPath = new Path(path, s"model-$idx").toString
          model.save(modelPath)
      }
      instance.subSpaces.zipWithIndex.foreach {
        case (subSpace, idx) =>
          val data = Data(subSpace)
          val dataPath = new Path(path, s"data-$idx").toString
          sparkSession.createDataFrame(Seq(data)).repartition(1).write.text(dataPath)
      }

    }
  }

  private class BaggingClassificationModelReader extends MLReader[BaggingClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BaggingClassificationModel].getName

    override def load(path: String): BaggingClassificationModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = BaggingClassifierParams.loadImpl(path, sc, className)
      val numModels = metadata.getParamValue("maxIter").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
      }
      val subSpaces = (0 until numModels).toArray.map { idx =>
        val dataPath = new Path(path, s"data-$idx").toString
        val data = sparkSession.read.text(dataPath).select("indices").head()
        data.getAs[Seq[Int]](0).toArray
      }
      val bcModel = new BaggingClassificationModel(metadata.uid, subSpaces, models)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
