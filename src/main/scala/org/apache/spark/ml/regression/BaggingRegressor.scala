package org.apache.spark.ml.regression

import org.apache.commons.math3.stat.StatUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.bagging.{Bagging, BaggingParams}
import org.apache.spark.ml.ensemble.{
  EnsemblePredictionModelType,
  EnsemblePredictorType,
  HasBaseLearner
}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql.Dataset
import org.apache.spark.util.ThreadUtils
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, JObject}

import scala.concurrent.Future
import scala.concurrent.duration.Duration

trait BaggingRegressorParams extends BaggingParams {}

object BaggingRegressorParams {

  def saveImpl(
      instance: BaggingRegressorParams,
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

class BaggingRegressor(override val uid: String)
    extends Predictor[Vector, BaggingRegressor, BaggingRegressionModel]
    with BaggingRegressorParams
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

  override def copy(extra: ParamMap): BaggingRegressor = {
    val copied = new BaggingRegressor(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  override protected def train(dataset: Dataset[_]): BaggingRegressionModel = instrumented {
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
          .transform(Bagging
            .withWeightedBag(getReplacement, getSampleRatio, getMaxIter, getSeed, "weightedBag"))

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

      new BaggingRegressionModel(models.toArray)

  }

  override def write: MLWriter = new BaggingRegressor.BaggingRegressorWriter(this)

}

object BaggingRegressor extends MLReadable[BaggingRegressor] {

  override def read: MLReader[BaggingRegressor] = new BaggingRegressorReader

  override def load(path: String): BaggingRegressor = super.load(path)

  private[BaggingRegressor] class BaggingRegressorWriter(instance: BaggingRegressor)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      BaggingRegressorParams.saveImpl(instance, path, sc)
    }

  }

  private class BaggingRegressorReader extends MLReader[BaggingRegressor] {

    /** Checked against metadata when loading model */
    private val className = classOf[BaggingRegressor].getName

    override def load(path: String): BaggingRegressor = {
      val (metadata, learner) = BaggingRegressorParams.loadImpl(path, sc, className)
      val bc = new BaggingRegressor(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

class BaggingRegressionModel(
    override val uid: String,
    val subSpaces: Array[Array[Int]],
    val models: Array[EnsemblePredictionModelType])
    extends RegressionModel[Vector, BaggingRegressionModel]
    with BaggingRegressorParams
    with MLWritable {

  def this(subSpaces: Array[Array[Int]], models: Array[EnsemblePredictionModelType]) =
    this(Identifiable.randomUID("BaggingRegressionModel"), subSpaces, models)

  def this(tuples: Array[(Array[Int], EnsemblePredictionModelType)]) =
    this(tuples.map(_._1), tuples.map(_._2))

  override def predict(features: Vector): Double = {
    StatUtils.mean(subSpaces.zip(models).map {
      case (subSpace, model) =>
        val subFeatures = features match {
          case features: DenseVector => Vectors.dense(subSpace.map(features.apply))
          case features: SparseVector => features.slice(subSpace)
        }
        model.predict(subFeatures)
    })
  }

  override def copy(extra: ParamMap): BaggingRegressionModel = {
    val copied = new BaggingRegressionModel(uid, subSpaces, models)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new BaggingRegressionModel.BaggingRegressionModelWriter(this)

}

object BaggingRegressionModel extends MLReadable[BaggingRegressionModel] {

  override def read: MLReader[BaggingRegressionModel] = new BaggingRegressionModelReader

  override def load(path: String): BaggingRegressionModel = super.load(path)

  private[BaggingRegressionModel] class BaggingRegressionModelWriter(
      instance: BaggingRegressionModel)
      extends MLWriter {

    private case class Data(subSpaces: Array[Int])

    override protected def saveImpl(path: String): Unit = {
      BaggingRegressorParams.saveImpl(instance, path, sc)
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

  private class BaggingRegressionModelReader extends MLReader[BaggingRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BaggingRegressionModel].getName

    override def load(path: String): BaggingRegressionModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = BaggingRegressorParams.loadImpl(path, sc, className)
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
      val bcModel = new BaggingRegressionModel(metadata.uid, subSpaces, models)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
