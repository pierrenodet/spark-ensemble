package org.apache.spark.ml.classification

import java.util.Locale

import breeze.linalg.DenseVector
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.boosting.GBMParams
import org.apache.spark.ml.ensemble._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.shared.{HasParallelism, HasWeightCol}
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair}
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.ThreadUtils
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{parse, render}
import org.json4s.{DefaultFormats, JObject}

import scala.annotation.tailrec
import scala.concurrent.duration.Duration
import scala.concurrent.{ExecutionContext, Future}

trait GBMClassifierParams extends ClassifierParams with GBMParams with HasParallelism {

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
        -y * breeze.numerics.log(prediction)
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
      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(
        this,
        labelCol,
        weightCol,
        featuresCol,
        predictionCol,
        loss,
        maxIter,
        learningRate,
        tol,
        seed)

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

      val labelSchema = dataset.schema($(labelCol))
      val computeNumClasses: () => Int = () => {
        val Row(maxLabelIndex: Double) =
          dataset.agg(max(col($(labelCol)).cast(DoubleType))).head()
        // classes are assumed to be numbered from 0,...,maxLabelIndex
        maxLabelIndex.toInt + 1
      }
      val numClasses =
        MetadataUtils.getNumClasses(labelSchema).fold(computeNumClasses())(identity)
      instr.logNumClasses(numClasses)

      @tailrec
      def trainBoosters(
          train: Dataset[_],
          labelColName: String,
          weightColName: Option[String],
          featuresColName: String,
          predictionColName: String,
          numClasses: Int,
          executionContext: ExecutionContext,
          baseLearner: EnsemblePredictorType,
          learningRate: Double,
          loss: (Double, Double) => Double,
          negGrad: (Double, Double) => Double,
          tol: Double,
          seed: Long,
          instrumentation: Instrumentation)(
          weights: Array[Array[Double]],
          boosters: Array[Array[EnsemblePredictionModelType]],
          iter: Int): (Array[Array[Double]], Array[Array[EnsemblePredictionModelType]]) = {

        if (iter == 0) {

          (weights, boosters)

        } else {

          instrumentation.logNamedValue("iteration", iter)

          val ngUDF = udf(negGrad)

          val current = new GBMClassificationModel(numClasses, weights, boosters)

          val predUDF = udf { features: Vector =>
            current.predictRaw(features).toArray
          }

          val newBoostersFutures = Range(0, numClasses).toArray
            .map(k => {
              val residuals = train.withColumn(
                labelColName,
                ngUDF(
                  when(col(labelColName) === k.toDouble, 1.0).otherwise(0.0),
                  predUDF(col(featuresColName))(k)),
                train.schema(train.schema.fieldIndex(labelColName)).metadata)
              val paramMap = new ParamMap()
              paramMap.put(baseLearner.labelCol -> labelColName)
              paramMap.put(baseLearner.featuresCol -> featuresColName)
              paramMap.put(baseLearner.predictionCol -> predictionColName)
              Future {
                if (weightColName.isDefined) {
                  val baseLearner_ =
                    baseLearner.asInstanceOf[EnsemblePredictorType with HasWeightCol]
                  paramMap.put(baseLearner_.weightCol -> weightColName.get)
                  baseLearner_.fit(residuals, paramMap)
                } else {
                  baseLearner.fit(residuals, paramMap)
                }
              }(executionContext)
            })

          val newBoosters = newBoostersFutures
            .map(ThreadUtils.awaitResult(_, Duration.Inf))
            .toArray[EnsemblePredictionModelType]

          instr.logNumFeatures(newBoosters.head.numFeatures)

          val newWeights = Array.fill(numClasses)(learningRate)

          instrumentation.logNamedValue("weight", newWeights)
          instrumentation.logInfo("boosters")
          newBoosters.foreach(instrumentation.logPipelineStage)

          trainBoosters(
            train,
            labelColName,
            weightColName,
            featuresColName,
            predictionColName,
            numClasses,
            executionContext,
            baseLearner,
            learningRate,
            loss,
            negGrad,
            tol,
            seed + iter,
            instrumentation)(weights :+ newWeights, boosters :+ newBoosters, iter - 1)
        }

      }

      val executionContext = getExecutionContext

      val initBoostersFutures = Range(0, numClasses).toArray
        .map(k => {
          val train = df.withColumn(
            $(labelCol),
            when(col($(labelCol)) === k.toDouble, 1.0).otherwise(0.0),
            df.schema(df.schema.fieldIndex(getLabelCol)).metadata)
          val baseLearner = getBaseLearner
          val paramMap = new ParamMap()
          paramMap.put(baseLearner.labelCol -> getLabelCol)
          paramMap.put(baseLearner.featuresCol -> getFeaturesCol)
          paramMap.put(baseLearner.predictionCol -> getPredictionCol)
          Future {
            if (weightColIsUsed) {
              val baseLearner_ = baseLearner.asInstanceOf[EnsemblePredictorType with HasWeightCol]
              paramMap.put(baseLearner_.weightCol -> getWeightCol)
              baseLearner_.fit(train, paramMap)
            } else {
              baseLearner.fit(train, paramMap)
            }
          }(executionContext)
        })

      val initBoosters = initBoostersFutures
        .map(ThreadUtils.awaitResult(_, Duration.Inf))
        .toArray[EnsemblePredictionModelType]

      instr.logNumFeatures(initBoosters.head.numFeatures)

      val initWeights = Array.fill(numClasses)(1.0)

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
          numClasses,
          executionContext,
          getBaseLearner,
          getLearningRate,
          GBMClassifierParams.lossFunction(getLoss),
          GBMClassifierParams.gradFunction(getLoss),
          getTol,
          getSeed,
          instr)(Array(initWeights), Array(initBoosters), getMaxIter)

      if (handlePersistence) {
        df.unpersist()
      }

      new GBMClassificationModel(numClasses, weights, boosters)

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

/* The models and weights are first indexed by the number of classes, then by the number of iterations*/
class GBMClassificationModel(
    override val uid: String,
    override val numClasses: Int,
    val weights: Array[Array[Double]],
    val models: Array[Array[EnsemblePredictionModelType]])
    extends ClassificationModel[Vector, GBMClassificationModel]
    with GBMClassifierParams
    with MLWritable {

  def this(
      numClasses: Int,
      weights: Array[Array[Double]],
      models: Array[Array[EnsemblePredictionModelType]]) =
    this(Identifiable.randomUID("BoostingRegressionModel"), numClasses, weights, models)

  override def predictRaw(features: Vector): Vector = {
    val kprediction = weights.zip(models).foldLeft(DenseVector.zeros[Double](numClasses)) {
      case (acc, (weight, model)) =>
        acc + DenseVector[Double](weight) * DenseVector[Double](model.map(_.predict(features)))
    }
    Vectors.fromBreeze(
      breeze.numerics.exp(kprediction) / breeze.linalg.sum(breeze.numerics.exp(kprediction)))
  }

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
      instance.models.zipWithIndex.foreach {
        case (models, idx) =>
          models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach {
            case (model, k) =>
              val modelPath = new Path(path, s"model-$k-$idx").toString
              model.save(modelPath)
          }
      }
      instance.weights.zipWithIndex.foreach {
        case (weights, idx) =>
          weights.zipWithIndex.foreach {
            case (weight, k) =>
              val data = Data(weight)
              val dataPath = new Path(path, s"data-$k-$idx").toString
              sparkSession.createDataFrame(Seq(data)).repartition(1).write.json(dataPath)
          }
      }

    }
  }

  private class GBMClassificationModelReader extends MLReader[GBMClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBMClassificationModel].getName

    override def load(path: String): GBMClassificationModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = GBMClassifierParams.loadImpl(path, sc, className)
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val numModels = metadata.getParamValue("maxIter").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        (0 until numClasses).map { k =>
          val modelPath = new Path(path, s"model-$k-$idx").toString
          DefaultParamsReader
            .loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
        }.toArray
      }
      val boostsData = (0 until numModels).toArray.map { idx =>
        (0 until numClasses).map { k =>
          val dataPath = new Path(path, s"data-$k-$idx").toString
          val data = sparkSession.read.json(dataPath).select("weight").head()
          data.getAs[Double](0)
        }.toArray
      }
      val bcModel =
        new GBMClassificationModel(metadata.uid, numClasses, boostsData, models)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
