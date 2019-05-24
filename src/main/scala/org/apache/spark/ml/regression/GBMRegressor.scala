package org.apache.spark.ml.regression

import java.util.Locale

import org.apache.commons.math3.util.FastMath
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.boosting.GBMParams
import org.apache.spark.ml.ensemble.{EnsemblePredictionModelType, EnsemblePredictorType, HasBaseLearner}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
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

      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(this, labelCol, weightCol, featuresCol, predictionCol,
        loss, maxIter, learningRate, tol, seed)

      val weightColIsUsed = isDefined(weightCol) && $(weightCol).nonEmpty && {
        getBaseLearner match {
          case _: HasWeightCol => true
          case c =>
            instr.logWarning(s"weightCol is ignored, as it is not supported by $c now.")
            false
        }
      }

      val df = if (weightColIsUsed) {
        dataset.select($(labelCol), $(featuresCol), $(weightCol))
      } else {
        dataset.select($(labelCol), $(featuresCol))
      }

      val handlePersistence = dataset.storageLevel == StorageLevel.NONE && (df.storageLevel == StorageLevel.NONE)
      if (handlePersistence) {
        df.persist(StorageLevel.MEMORY_AND_DISK)
      }

      @tailrec
      def trainBoosters(
                         train: Dataset[_],
                         labelColName: String,
                         weightColName: Option[String],
                         featuresColName: String,
                         predictionColName: String,
                         baseLearner: EnsemblePredictorType,
                         learningRate: Double,
                         loss: (Double, Double) => Double,
                         negGrad: (Double, Double) => Double,
                         tol: Double,
                         seed: Long)(
                         weights: Array[Double],
                         boosters: Array[EnsemblePredictionModelType],
                         iter: Int): (Array[Double], Array[EnsemblePredictionModelType]) = {

        if (iter == 0) {

          (weights, boosters)

        } else {

          instr.logNamedValue("iteration", iter)

          val ngUDF = udf(negGrad)

          val current = new GBMRegressionModel(weights, boosters)

          val predUDF = udf { features: Vector => current.predict(features) }

          val residuals = train
            .withColumn(labelColName, ngUDF(col(labelColName), predUDF(col(featuresColName))), train.schema(train.schema.fieldIndex(labelColName)).metadata)

          val paramMap = new ParamMap()
          paramMap.put(baseLearner.labelCol -> labelColName)
          paramMap.put(baseLearner.featuresCol -> featuresColName)
          paramMap.put(baseLearner.predictionCol -> predictionColName)

          residuals.schema.foreach(kek => println(kek.metadata.toString()))
          println(learningRate)
          println(iter)
          residuals.show()

          val booster = if (weightColName.isDefined) {
            val baseLearner_ = baseLearner.asInstanceOf[EnsemblePredictorType with HasWeightCol]
            paramMap.put(baseLearner_.weightCol -> weightColName.get)
            baseLearner_.fit(residuals, paramMap)
          } else {
            baseLearner.fit(residuals, paramMap)
          }

          val weight = learningRate

          instr.logNamedValue("weight", weight)
          instr.logInfo("booster")
          instr.logPipelineStage(booster)

          trainBoosters(train, labelColName, weightColName, featuresColName, predictionColName, baseLearner, learningRate, loss, negGrad, tol, seed + iter)(
            weights :+ weight,
            boosters :+ booster.asInstanceOf[EnsemblePredictionModelType],
            iter - 1)
        }

      }

      val paramMap = new ParamMap()
      paramMap.put(getBaseLearner.labelCol -> getLabelCol)
      paramMap.put(getBaseLearner.featuresCol -> getFeaturesCol)
      paramMap.put(getBaseLearner.predictionCol -> getPredictionCol)

      val initBooster = if (weightColIsUsed) {
        val baseLearner_ = getBaseLearner.asInstanceOf[EnsemblePredictorType with HasWeightCol]
        paramMap.put(baseLearner_.weightCol -> getWeightCol)
        baseLearner_.fit(df, paramMap)
      } else {
        getBaseLearner.fit(df, paramMap)
      }

      val initWeight = 1

      val optWeightColName = if (weightColIsUsed) {
        Some($(weightCol))
      } else {
        None
      }

      val (weights, boosters) =
        trainBoosters(
          df,
          getLabelCol,
          optWeightColName,
          getFeaturesCol,
          getPredictionCol,
          getBaseLearner,
          getLearningRate,
          GBMRegressorParams.lossFunction(getLoss),
          GBMRegressorParams.gradFunction(getLoss),
          getTol,
          getSeed
        )(Array(initWeight), Array(initBooster.asInstanceOf[EnsemblePredictionModelType]), getMaxIter)

      if (handlePersistence) {
        df.unpersist()
      }

      new GBMRegressionModel(weights, boosters)

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
