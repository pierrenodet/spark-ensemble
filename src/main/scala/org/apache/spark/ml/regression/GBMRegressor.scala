package org.apache.spark.ml.regression

import java.util.Locale

import breeze.linalg.DenseVector
import breeze.optimize.{ApproximateGradientFunction, LBFGS}
import org.apache.commons.math3.util.FastMath
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.boosting.GBMParams
import org.apache.spark.ml.ensemble.{
  EnsemblePredictionModelType,
  EnsemblePredictorType,
  HasBaseLearner
}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{parse, render}
import org.json4s.{DefaultFormats, JObject}

import scala.annotation.tailrec

trait GBMRegressorParams extends GBMParams {

  /**
   * Loss function which Boosting tries to minimize. (case-insensitive)
   * Supported: "ls", "lad", "huber", "quantile".
   * (default = ls)
   *
   * @group param
   */
  val loss: Param[String] =
    new Param(
      this,
      "loss",
      "loss function, (case-insensitive). Supported options:" + s"${GBMRegressorParams.supportedLossTypes
        .mkString(",")}",
      (value: String) =>
        GBMRegressorParams.supportedLossTypes.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getLoss: String = $(loss).toLowerCase(Locale.ROOT)

  setDefault(loss -> "ls")

}

object GBMRegressorParams {

  final val supportedLossTypes: Array[String] =
    Array("ls", "lad").map(_.toLowerCase(Locale.ROOT))

  def lossFunction(loss: String): (Double, Double) => Double = loss match {
    case "ls" =>
      (y, prediction) =>
        FastMath.pow(y - prediction, 2) / 2
    case "lad" =>
      (y, prediction) =>
        FastMath.abs(y - prediction)
    case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")
  }

  def gradFunction(loss: String): (Double, Double) => Double = loss match {
    case "ls" =>
      (y, prediction) =>
        y - prediction
    case "lad" =>
      (y, prediction) =>
        FastMath.signum(y - prediction)
    case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")
  }

  def saveImpl(
      instance: GBMRegressorParams,
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

class GBMRegressor(override val uid: String)
    extends Predictor[Vector, GBMRegressor, GBMRegressionModel]
    with GBMRegressorParams
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

  /** @group setParam */
  def setTol(value: Double): this.type = set(tol, value)

  def this() = this(Identifiable.randomUID("GBMRegressor"))

  override def copy(extra: ParamMap): GBMRegressor = {
    val copied = new GBMRegressor(uid)
    copyValues(copied, extra)
    copied.setBaseLearner(copied.getBaseLearner.copy(extra))
  }

  override protected def train(dataset: Dataset[_]): GBMRegressionModel =
    instrumented { instr =>
      val spark = dataset.sparkSession

      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(this, maxIter, seed)

      val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
      if (!isDefined(weightCol) || $(weightCol).isEmpty) setWeightCol("weight")

      val instances: RDD[Instance] =
        dataset.select(col($(labelCol)), w, col($(featuresCol))).rdd.map {
          case Row(label: Double, weight: Double, features: Vector) =>
            Instance(label, weight, features)
        }

      val persistedInput = if (instances.getStorageLevel == StorageLevel.NONE) {
        instances.persist(StorageLevel.MEMORY_AND_DISK)
        true
      } else {
        false
      }

      val lossFunction: (Double, Double) => Double =
        GBMRegressorParams.lossFunction(getLoss)
      val negGradFunction: (Double, Double) => Double =
        GBMRegressorParams.gradFunction(getLoss)

      def trainBooster(
          baseLearner: EnsemblePredictorType,
          learningRate: Double,
          tol: Double,
          seed: Long,
          loss: (Double, Double) => Double,
          negGrad: (Double, Double) => Double,
          weights: Array[Double],
          boosters: Array[EnsemblePredictionModelType])(
          instances: RDD[Instance]): (Double, EnsemblePredictionModelType) = {

        val weightedBoosters = weights.zip(boosters)

        val residuals = instances.map(instance =>
          Instance(negGrad(instance.label, weightedBoosters.map {
            case (weight, model) => weight * model.predict(instance.features)
          }.sum), instance.weight, instance.features))

        val residualsDF =
          spark.createDataFrame(residuals)

        val paramMap = new ParamMap()
        paramMap.put(getBaseLearner.labelCol -> "label")
        paramMap.put(getBaseLearner.featuresCol -> "features")

        val booster = getBaseLearner.fit(residualsDF, paramMap)

        /* def weightFunction(instances: RDD[Instance])(x: DenseVector[Double]): Double = {
          instances
            .map(instance =>
              loss(instance.label, weightedBoosters.map {
                case (weight, model) => weight * model.predict(instance.features)
              }.sum + x(0) * booster.predict(instance.features)))
            .sum
        }

        val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 100000, m = 7, tolerance = tol)

        val agf = new ApproximateGradientFunction(weightFunction(instances))

        val pho = lbfgs.minimize(agf, DenseVector(0.1))

        (pho(0) * */
        (learningRate, booster.asInstanceOf[EnsemblePredictionModelType])

      }

      @tailrec
      def trainBoosters(
          baseLearner: EnsemblePredictorType,
          learningRate: Double,
          tol: Double,
          seed: Long,
          loss: (Double, Double) => Double,
          negGrad: (Double, Double) => Double)(
          instances: RDD[Instance],
          weights: Array[Double],
          boosters: Array[EnsemblePredictionModelType],
          iter: Int): (Array[Double], Array[EnsemblePredictionModelType]) = {

        val (weight, booster) =
          trainBooster(
            baseLearner,
            learningRate,
            tol,
            seed + iter,
            loss,
            negGrad,
            weights,
            boosters)(instances)

        if (iter == 0) {
          (weights, boosters)
        } else {
          trainBoosters(baseLearner, learningRate, tol, seed + iter, loss, negGrad)(
            instances,
            weights :+ weight,
            boosters :+ booster,
            iter - 1)
        }
      }

      val df =
        spark.createDataFrame(instances)

      val paramMap = new ParamMap()
      paramMap.put(getBaseLearner.labelCol -> "label")
      paramMap.put(getBaseLearner.featuresCol -> "features")

      val init = getBaseLearner.fit(df, paramMap).asInstanceOf[EnsemblePredictionModelType]

      val boosters =
        trainBoosters(
          getBaseLearner,
          getLearningRate,
          getTol,
          getSeed,
          lossFunction,
          negGradFunction)(instances, Array(1), Array(init), getMaxIter)

      if (persistedInput) instances.unpersist()

      new GBMRegressionModel(boosters._1, boosters._2)

    }

  override def write: MLWriter =
    new GBMRegressor.GBMRegressorWriter(this)

}

object GBMRegressor extends MLReadable[GBMRegressor] {

  override def read: MLReader[GBMRegressor] = new GBMRegressorReader

  override def load(path: String): GBMRegressor = super.load(path)

  private[GBMRegressor] class GBMRegressorWriter(instance: GBMRegressor) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      GBMRegressorParams.saveImpl(instance, path, sc)
    }

  }

  private class GBMRegressorReader extends MLReader[GBMRegressor] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBMRegressor].getName

    override def load(path: String): GBMRegressor = {
      val (metadata, learner) = GBMRegressorParams.loadImpl(path, sc, className)
      val bc = new GBMRegressor(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

class GBMRegressionModel(
    override val uid: String,
    val weights: Array[Double],
    val models: Array[EnsemblePredictionModelType])
    extends PredictionModel[Vector, GBMRegressionModel]
    with GBMRegressorParams
    with MLWritable {

  def this(weights: Array[Double], models: Array[EnsemblePredictionModelType]) =
    this(Identifiable.randomUID("BoostingRegressionModel"), weights, models)

  def this(tuples: Array[(Double, EnsemblePredictionModelType)]) =
    this(Identifiable.randomUID("BoostingRegressionModel"), tuples.map(_._1), tuples.map(_._2))

  override def predict(features: Vector): Double =
    weights
      .zip(models)
      .map {
        case (weight, model) =>
          weight * model.predict(features)
      }
      .sum

  override def copy(extra: ParamMap): GBMRegressionModel = {
    val copied = new GBMRegressionModel(uid, weights, models)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter =
    new GBMRegressionModel.GBMRegressionModelWriter(this)

}

object GBMRegressionModel extends MLReadable[GBMRegressionModel] {

  override def read: MLReader[GBMRegressionModel] =
    new GBMRegressionModelReader

  override def load(path: String): GBMRegressionModel = super.load(path)

  private[GBMRegressionModel] class GBMRegressionModelWriter(instance: GBMRegressionModel)
      extends MLWriter {

    private case class Data(weight: Double)

    override protected def saveImpl(path: String): Unit = {
      GBMRegressorParams.saveImpl(instance, path, sc)
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

  private class GBMRegressionModelReader extends MLReader[GBMRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBMRegressionModel].getName

    override def load(path: String): GBMRegressionModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = GBMRegressorParams.loadImpl(path, sc, className)
      val numModels = metadata.getParamValue("maxIter").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
      }
      val boostsData = (0 until numModels).toArray.map { idx =>
        val dataPath = new Path(path, s"data-$idx").toString
        val data = sparkSession.read.json(dataPath).select("weight").head()
        data.getAs[Double](0)
      }
      val bcModel =
        new GBMRegressionModel(metadata.uid, boostsData, models)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
