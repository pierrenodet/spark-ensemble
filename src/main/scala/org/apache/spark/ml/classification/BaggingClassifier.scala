package org.apache.spark.ml.classification

import org.apache.commons.math3.stat.StatUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.bagging._
import org.apache.spark.ml.ensemble.{EnsemblePredictionModelType, EnsemblePredictorType, HasBaseLearner}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel
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

  def this() = this(Identifiable.randomUID("BaggingClassifier"))

  // Parameters from BaggingClassifierParams:

  /** @group setParam */
  def setBaseLearner(value: Predictor[_, _, _]): this.type =
    set(baseLearner, value.asInstanceOf[EnsemblePredictorType])

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

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
      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(this, maxIter, seed, parallelism)

      val spark = dataset.sparkSession

      val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
      if (!isDefined(weightCol) || $(weightCol).isEmpty) setWeightCol("weight")

      val input: RDD[Instance] =
        dataset
          .select(col($(labelCol)), w, col($(featuresCol)))
          .rdd
          .map {
            case Row(label: Double, weight: Double, features: Vector) =>
              Instance(label, weight, features)
          }
          .persist(StorageLevel.MEMORY_AND_DISK)

      val numFeatures = input.first().features.size

      val baggedInput = BaggedPoint
        .convertToBaggedRDD(
          input,
          getSampleRatio,
          getMaxIter,
          getReplacement,
          (instance: Instance) => instance.weight,
          getSeed)

      val futureModels = (0 until getMaxIter).map(iter =>
        Future[Option[(Array[Int], EnsemblePredictionModelType)]] {

          val sampled =
            baggedInput.flatMap(bi => Iterator.fill(bi.subsampleCounts(iter))(bi.datum))

          val patch =
            PatchedPoint
              .patch(getSampleRatioFeatures, numFeatures, getReplacementFeatures, getSeed + iter)

          if (patch sameElements Array.fill(numFeatures)(0.0)) {

            //Nothing to learn as no features have been chosen
            None

          } else {

            val subspaced = PatchedPoint.convertToPatchedRDD(sampled, patch)

            val df = spark.createDataFrame(subspaced)
            instr.logDebug(s"Start training for $iter iteration on $df with $getBaseLearner")

            val paramMap = new ParamMap()
            paramMap.put(getBaseLearner.labelCol -> "label")
            paramMap.put(getBaseLearner.featuresCol -> "features")

            val model = getBaseLearner.fit(df,paramMap)

            instr.logDebug(s"Training done for $iter iteration on $df with $getBaseLearner")

            Some(patch, model)

          }

        }(getExecutionContext))

      val models = futureModels.map(f => ThreadUtils.awaitResult(f, Duration.Inf))

      input.unpersist()

      new BaggingClassificationModel(models.flatten.toArray)

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
    StatUtils
      .mode(subSpaces.zip(models).map {
        case (subSpace, model) =>
          val subFeatures =
            Vectors.dense(features.toArray.zip(subSpace).flatMap {
              case (f, i) => if (i == 0) None else Some(f)
            })
          model.predict(subFeatures)
      })
      .head

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

    private case class Data(subSpace: Array[Int])

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
          sparkSession.createDataFrame(Seq(data)).repartition(1).write.json(dataPath)
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
        val data = sparkSession.read.json(dataPath).select("subSpace").head()
        data.getAs[Seq[Int]](0).toArray
      }
      val bcModel = new BaggingClassificationModel(metadata.uid, subSpaces, models)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
