package org.apache.spark.ml.classification

import java.util.Locale

import breeze.linalg.DenseVector
import breeze.numerics._
import breeze.optimize.{ApproximateGradientFunction, LBFGS}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.boosting.GBMParams
import org.apache.spark.ml.ensemble._
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

import scala.annotation.tailrec

trait GBMClassifierParams extends ClassifierParams with GBMParams {

  /**
   * Loss function which Boosting tries to minimize. (case-insensitive)
   * Supported: "lk".
   * (default = lk)
   *
   * @group param
   */
  val loss: Param[String] =
    new Param(
      this,
      "loss",
      "loss function, (case-insensitive). Supported options:" + s"${GBMClassifierParams.supportedLossTypes
        .mkString(",")}",
      (value: String) =>
        GBMClassifierParams.supportedLossTypes.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getLoss: String = $(loss).toLowerCase(Locale.ROOT)

  setDefault(loss -> "lk")

}

object GBMClassifierParams {

  final val supportedLossTypes: Array[String] =
    Array("lk").map(_.toLowerCase(Locale.ROOT))

  def lossFunction(loss: String): (Double, Double) => Double = loss match {
    case "lk" =>
      (y, prediction) =>
        -log(prediction)
    case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")
  }

  def gradFunction(loss: String): (Double, Double) => Double = loss match {
    case "lk" =>
      (y, prediction) =>
        y - prediction
    case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")
  }

  def saveImpl(
      instance: GBMClassifierParams,
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

  def loadImpl(path: String, sc: SparkContext, expectedClassName: String)
    : (DefaultParamsReader.Metadata, EnsembleProbabilisticClassifierType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val learner = HasBaseProbabilisticClassifier.loadImpl(path, sc)
    (metadata, learner)
  }

}

class GBMClassifier(override val uid: String)
    extends Classifier[Vector, GBMClassifier, GBMClassificationModel]
    with GBMClassifierParams
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

  def this() = this(Identifiable.randomUID("GBMClassifier"))

  override def copy(extra: ParamMap): GBMClassifier = {
    val copied = new GBMClassifier(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  override protected def train(dataset: Dataset[_]): GBMClassificationModel =
    instrumented { instr =>
      val spark = dataset.sparkSession

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

      val lossFunction: (Double, Double) => Double =
        GBMClassifierParams.lossFunction(getLoss)
      val negGradFunction: (Double, Double) => Double =
        GBMClassifierParams.gradFunction(getLoss)

      def trainBooster(
          baseLearner: EnsemblePredictorType,
          learningRate: Double,
          seed: Long,
          loss: (Double, Double) => Double,
          negGrad: (Double, Double) => Double,
          weights: Array[Double],
          boosters: Array[EnsemblePredictionModelType])(
          instances: RDD[Instance]): (Double, EnsemblePredictionModelType) = {

        val weightedBoosters = weights.zip(boosters)

        val losses = instances.map(instance =>
          negGrad(instance.label, weightedBoosters.map {
            case (weight, model) => weight * model.predict(instance.features)
          }.sum))

        val df =
          spark.createDataFrame(instances.zip(losses).map {
            case (instance, loss) => Instance(loss, instance.weight, instance.features)
          })

        val paramMap = new ParamMap()
        paramMap.put(getBaseLearner.labelCol -> "label")
        paramMap.put(getBaseLearner.featuresCol -> "features")

        val booster = getBaseLearner.fit(df, paramMap)

        def weightFunction(instances: RDD[Instance])(x: DenseVector[Double]): Double = {
          instances
            .map(instance =>
              loss(instance.label, weightedBoosters.map {
                case (weight, model) => weight * model.predict(instance.features)
              }.sum + x(0) * booster.predict(instance.features)))
            .map(loss => breeze.numerics.pow(loss, 2))
            .sum
        }

        val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 100, m = 3)

        val agf = new ApproximateGradientFunction(weightFunction(instances))

        val pho = lbfgs.minimize(agf, DenseVector(0))

        (pho(0) * learningRate, booster.asInstanceOf[EnsemblePredictionModelType])

      }

      @tailrec
      def trainBoosters(
          baseLearner: EnsemblePredictorType,
          learningRate: Double,
          seed: Long,
          loss: (Double, Double) => Double,
          negGrad: (Double, Double) => Double)(
          instances: RDD[Instance],
          weights: Array[Double],
          boosters: Array[EnsemblePredictionModelType],
          iter: Int): (Array[Double], Array[EnsemblePredictionModelType]) = {

        val persistedInput = if (instances.getStorageLevel == StorageLevel.NONE) {
          instances.persist(StorageLevel.MEMORY_AND_DISK)
          true
        } else {
          false
        }

        val (weight, booster) =
          trainBooster(baseLearner, learningRate, seed + iter, loss, negGrad, weights, boosters)(
            instances)

        if (iter == 0) {
          if (persistedInput) instances.unpersist()
          (weights, boosters)
        } else {
          trainBoosters(baseLearner, learningRate, seed + iter, loss, negGrad)(
            instances,
            weights :+ weight,
            boosters :+ booster,
            iter - 1)
        }
      }

      val boosters =
        trainBoosters(getBaseLearner, getLearningRate, getSeed, lossFunction, negGradFunction)(
          instances,
          Array.empty[Double],
          Array.empty[EnsemblePredictionModelType],
          getMaxIter)

      new GBMClassificationModel(numClasses, boosters._1, boosters._2)

    }

  override def write: MLWriter =
    new GBMClassifier.GBMClassifierWriter(this)

}

object GBMClassifier extends MLReadable[GBMClassifier] {

  override def read: MLReader[GBMClassifier] = new GBMClassifierReader

  override def load(path: String): GBMClassifier = super.load(path)

  private[GBMClassifier] class GBMClassifierWriter(instance: GBMClassifier) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      GBMClassifierParams.saveImpl(instance, path, sc)
    }

  }

  private class GBMClassifierReader extends MLReader[GBMClassifier] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBMClassifier].getName

    override def load(path: String): GBMClassifier = {
      val (metadata, learner) = GBMClassifierParams.loadImpl(path, sc, className)
      val bc = new GBMClassifier(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

class GBMClassificationModel(
    override val uid: String,
    override val numClasses: Int,
    val weights: Array[Double],
    val models: Array[EnsemblePredictionModelType])
    extends ClassificationModel[Vector, GBMClassificationModel]
    with GBMClassifierParams
    with MLWritable {

  def this(numClasses: Int, weights: Array[Double], models: Array[EnsemblePredictionModelType]) =
    this(Identifiable.randomUID("BoostingRegressionModel"), numClasses, weights, models)

  def this(numClasses: Int, tuples: Array[(Double, EnsemblePredictionModelType)]) =
    this(
      Identifiable.randomUID("BoostingRegressionModel"),
      numClasses,
      tuples.map(_._1),
      tuples.map(_._2))

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

  override def copy(extra: ParamMap): GBMClassificationModel = {
    val copied = new GBMClassificationModel(uid, numClasses, weights, models)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter =
    new GBMClassificationModel.GBMClassificationModelWriter(this)

}

object GBMClassificationModel extends MLReadable[GBMClassificationModel] {

  override def read: MLReader[GBMClassificationModel] =
    new GBMClassificationModelReader

  override def load(path: String): GBMClassificationModel = super.load(path)

  private[GBMClassificationModel] class GBMClassificationModelWriter(
      instance: GBMClassificationModel)
      extends MLWriter {

    private case class Data(weight: Double)

    override protected def saveImpl(path: String): Unit = {
      val extraJson = "numClasses" -> instance.numClasses
      GBMClassifierParams.saveImpl(instance, path, sc, Some(extraJson))
      instance.models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach {
        case (model, idx) =>
          val modelPath = new Path(path, s"model-$idx").toString
          model.save(modelPath)
      }
      instance.weights.zipWithIndex.foreach {
        case (weight, idx) =>
          val data = Data(weight)
          val dataPath = new Path(path, s"data-$idx").toString
          sparkSession.createDataFrame(Seq(data)).repartition(1).write.json(dataPath)
      }

    }
  }

  private class GBMClassificationModelReader extends MLReader[GBMClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBMClassificationModel].getName

    override def load(path: String): GBMClassificationModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = GBMClassifierParams.loadImpl(path, sc, className)
      val numModels = metadata.getParamValue("maxIter").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader
          .loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
      }
      val boostsData = (0 until numModels).toArray.map { idx =>
        val dataPath = new Path(path, s"data-$idx").toString
        val data = sparkSession.read.json(dataPath).select("weight").head()
        data.getAs[Double](0)
      }
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val bcModel =
        new GBMClassificationModel(metadata.uid, numClasses, boostsData, models)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
