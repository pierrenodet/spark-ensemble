package org.apache.spark.ml.classification
import java.util.Locale

import breeze.linalg.DenseVector
import org.apache.commons.math3.distribution.PoissonDistribution
import org.apache.commons.math3.util.FastMath
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.boosting.BoostingParams
import org.apache.spark.ml.ensemble.{EnsemblePredictionModelType, EnsemblePredictorType, HasBaseLearner}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{parse, render}
import org.json4s.{DefaultFormats, JObject}

trait BoostingClassifierParams extends BoostingParams with ClassifierParams {

  /**
   * Loss function which Boosting tries to minimize. (case-insensitive)
   * Supported: "exponential"
   * (default = exponential)
   *
   * @group param
   */
  val loss: Param[String] =
    new Param(
      this,
      "loss",
      "loss function, exponential by default",
      (value: String) =>
        BoostingClassifierParams.supportedLossTypes.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getLoss: String = $(loss).toLowerCase(Locale.ROOT)

  setDefault(loss -> "exponential")

}

object BoostingClassifierParams {

  final val supportedLossTypes: Array[String] =
    Array("exponential", "squared", "absolute").map(_.toLowerCase(Locale.ROOT))

  def lossFunction(loss: String): Double => Double = loss match {
    case "exponential" =>
      error =>
        1 - FastMath.exp(-error)
    case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")

  }

  def saveImpl(
      instance: BoostingClassifierParams,
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

class BoostingClassifier(override val uid: String)
    extends Classifier[Vector, BoostingClassifier, BoostingClassificationModel]
    with BoostingClassifierParams
    with MLWritable {

  def setBaseLearner(value: Predictor[_, _, _]): this.type =
    set(baseLearner, value.asInstanceOf[EnsemblePredictorType])

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /** @group setParam */
  def setLoss(value: String): this.type = set(loss, value)

  def this() = this(Identifiable.randomUID("BoostingRegressor"))

  override def copy(extra: ParamMap): BoostingClassifier = {
    val copied = new BoostingClassifier(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  override protected def train(dataset: Dataset[_]): BoostingClassificationModel = instrumented {
    instr =>
      val spark = dataset.sparkSession

      val classifier = getBaseLearner
      setBaseLearner(
        classifier
          .set(classifier.labelCol, getLabelCol)
          .set(classifier.featuresCol, getFeaturesCol)
          .set(classifier.predictionCol, getPredictionCol))

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

      val lossFunction: Double => Double = BoostingClassifierParams.lossFunction(getLoss)

      def trainBooster(
                        baseLearner: EnsemblePredictorType,
                        learningRate: Double,
                        seed: Long,
                        loss: Double => Double)(instances: RDD[Instance])
      : (Option[(Double, EnsemblePredictionModelType)], RDD[Instance]) = {

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
          return (None, instances)
        }

        val sampledDF =
          spark.createDataFrame(sampled).toDF(labelColName, "weight", featuresColName)
        val model = baseLearner.fit(sampledDF)

        //TODO: Implement multiclass loss function
        val errors = instances.map {
          case Instance(label, _, features) => if (model.predict(features) != label) 1 else 0
        }
        val estimatorError =
          normalized
            .map(_.weight)
            .zip(errors)
            .map { case (weight, error) => weight * error }
            .reduce(_ + _)

        if (estimatorError <= 0) {
          return (Some(1, model), instances)
        }

        val beta = estimatorError / (1 - estimatorError)
        val estimatorWeight = learningRate * (FastMath.log(1 / beta) + FastMath.log(
          numClasses - 1))
        val instancesWithNewWeights = instances.zip(errors).map {
          case (Instance(label, weight, features), error) =>
            Instance(label, weight * FastMath.exp(estimatorWeight * error), features)
        }
        (Some(estimatorWeight, model), instancesWithNewWeights)

      }

      def trainBoosters(
                         baseLearner: EnsemblePredictorType,
                         learningRate: Double,
                         seed: Long,
                         loss: Double => Double)(
                         instances: RDD[Instance],
                         acc: Array[Option[(Double, EnsemblePredictionModelType)]],
                         iter: Int): Array[Option[(Double, EnsemblePredictionModelType)]] = {

        val persistedInput = if (instances.getStorageLevel == StorageLevel.NONE) {
          instances.persist(StorageLevel.MEMORY_AND_DISK)
          true
        } else {
          false
        }

        val (bpm, updated) =
          trainBooster(baseLearner, learningRate, seed + iter, loss)(instances)

        if (iter == 0) {
          if (persistedInput) instances.unpersist()
          acc
        } else {
          trainBoosters(baseLearner, learningRate, seed + iter, loss)(
            updated,
            acc ++ Array(bpm),
            iter - 1)
        }
      }

      val models =
        trainBoosters(getBaseLearner, getLearningRate, getSeed, lossFunction)(
          instances,
          Array.empty,
          getMaxIter)

      val usefulModels = models.flatten

      new BoostingClassificationModel(numClasses, usefulModels)

  }

  override def write: MLWriter = new BoostingClassifier.BoostingClassifierWriter(this)

}

object BoostingClassifier extends MLReadable[BoostingClassifier] {

  override def read: MLReader[BoostingClassifier] = new BoostingClassifierReader

  override def load(path: String): BoostingClassifier = super.load(path)

  private[BoostingClassifier] class BoostingClassifierWriter(instance: BoostingClassifier)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      BoostingClassifierParams.saveImpl(instance, path, sc)
    }

  }

  private class BoostingClassifierReader extends MLReader[BoostingClassifier] {

    /** Checked against metadata when loading model */
    private val className = classOf[BoostingClassifier].getName

    override def load(path: String): BoostingClassifier = {
      val (metadata, learner) = BoostingClassifierParams.loadImpl(path, sc, className)
      val bc = new BoostingClassifier(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

class BoostingClassificationModel(
    override val uid: String,
    override val numClasses: Int,
    val weights: Array[Double],
    val models: Array[EnsemblePredictionModelType])
    extends ClassificationModel[Vector, BoostingClassificationModel]
    with BoostingClassifierParams
    with MLWritable {

  def this(numClasses: Int, weights: Array[Double], models: Array[EnsemblePredictionModelType]) =
    this(Identifiable.randomUID("BoostingRegressionModel"), numClasses, weights, models)

  def this(numClasses: Int, tuples: Array[(Double, EnsemblePredictionModelType)]) =
    this(
      Identifiable.randomUID("BoostingRegressionModel"),
      numClasses,
      tuples.unzip._1,
      tuples.unzip._2)

  override protected def predictRaw(features: Vector): Vector =
    Vectors.fromBreeze(
      weights
        .zip(models)
        .map {
          case (weight, model) =>
            val tmp = DenseVector.zeros[Double](numClasses)
            tmp(model.predict(features).ceil.toInt) = 1.0
            val res = weight * tmp
            res
        }
        .reduce(_ + _))

  override def copy(extra: ParamMap): BoostingClassificationModel = {
    val copied = new BoostingClassificationModel(uid, numClasses, weights, models)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter =
    new BoostingClassificationModel.BoostingClassificationModelWriter(this)

}

object BoostingClassificationModel extends MLReadable[BoostingClassificationModel] {

  override def read: MLReader[BoostingClassificationModel] = new BoostingClassificationModelReader

  override def load(path: String): BoostingClassificationModel = super.load(path)

  private[BoostingClassificationModel] class BoostingClassificationModelWriter(
      instance: BoostingClassificationModel)
      extends MLWriter {

    private case class Data(weight: Double)

    override protected def saveImpl(path: String): Unit = {
      val extraJson = "numClasses" -> instance.numClasses
      BoostingClassifierParams.saveImpl(instance, path, sc, Some(extraJson))
      instance.models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach {
        case (model, idx) =>
          val modelPath = new Path(path, s"model-$idx").toString
          model.save(modelPath)
      }
      instance.weights.zipWithIndex.foreach {
        case (weight, idx) =>
          val data = Data(weight)
          val dataPath = new Path(path, s"data-$idx").toString
          sparkSession.createDataFrame(Seq(data)).repartition(1).write.text(dataPath)
      }

    }
  }

  private class BoostingClassificationModelReader extends MLReader[BoostingClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BoostingClassificationModel].getName

    override def load(path: String): BoostingClassificationModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = BoostingClassifierParams.loadImpl(path, sc, className)
      val numModels = metadata.getParamValue("maxIter").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
      }
      val boostsData = (0 until numModels).toArray.map { idx =>
        val dataPath = new Path(path, s"data-$idx").toString
        val data = sparkSession.read.text(dataPath).select("weight").head()
        data.getAs[Double](0)
      }
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val bcModel =
        new BoostingClassificationModel(metadata.uid, numClasses, boostsData, models)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
