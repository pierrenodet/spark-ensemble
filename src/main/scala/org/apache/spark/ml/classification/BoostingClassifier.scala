package org.apache.spark.ml.classification
import breeze.linalg.DenseVector
import org.apache.commons.math3.distribution.PoissonDistribution
import org.apache.commons.math3.util.FastMath
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.bagging.{BaggingParams, BaggingPredictor}
import org.apache.spark.ml.boosting.{BoostedPredictionModel, BoostingParams}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor, classification}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._

trait BoostingClassifierParams extends BoostingParams with ClassifierParams {}

class BoostingClassifier(override val uid: String)
    extends Classifier[Vector, BoostingClassifier, BoostingClassificationModel]
    with BoostingClassifierParams
    with MLWritable {

  def setBaseLearner(value: Predictor[_, _, _]): this.type =
    set(baseLearner, value.asInstanceOf[PredictorVectorType])

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

      val lossFunction: Double => Double = BoostingParams.lossFunction(getLoss)

      def trainBooster(
          baseLearner: PredictorVectorType,
          learningRate: Double,
          seed: Long,
          loss: Double => Double)(
          instances: RDD[Instance]): (BoostedPredictionModel, RDD[Instance]) = {

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
          val bpm = new BoostedPredictionModel(1, 0, null)
          return (bpm, instances)
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
          val bpm = new BoostedPredictionModel(0, 1, model)
          return (bpm, instances)
        }

        val beta = estimatorError / (1 - estimatorError)
        val estimatorWeight = learningRate * (FastMath.log(1 / beta) + FastMath.log(
          numClasses - 1))
        val instancesWithNewWeights = instances.zip(errors).map {
          case (Instance(label, weight, features), error) =>
            Instance(label, weight * FastMath.exp(estimatorWeight * error), features)
        }
        val bpm = new BoostedPredictionModel(estimatorError, estimatorWeight, model)
        (bpm, instancesWithNewWeights)

      }

      def trainBoosters(
          baseLearner: PredictorVectorType,
          learningRate: Double,
          seed: Long,
          loss: Double => Double)(
          instances: RDD[Instance],
          acc: Array[BoostedPredictionModel],
          iter: Int): Array[BoostedPredictionModel] = {

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

      val usefulModels = models.filter(_.weight > 0)

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
      BoostingParams.saveImpl(path, instance, sc)
    }

  }

  private class BoostingClassifierReader extends MLReader[BoostingClassifier] {

    /** Checked against metadata when loading model */
    private val className = classOf[BoostingClassifier].getName

    override def load(path: String): BoostingClassifier = {
      val (metadata, learner) = BoostingParams.loadImpl(path, sc, className)
      val bc = new BoostingClassifier(metadata.uid)
      metadata.getAndSetParams(bc)
      bc.setBaseLearner(learner)
    }
  }

}

class BoostingClassificationModel(
    override val uid: String,
    override val numClasses: Int,
    val models: Array[BoostedPredictionModel])
    extends ClassificationModel[Vector, BoostingClassificationModel]
    with BoostingClassifierParams
    with MLWritable {

  def this(numClasses: Int, models: Array[BoostedPredictionModel]) =
    this(Identifiable.randomUID("BoostingRegressionModel"), numClasses, models)

  override protected def predictRaw(features: Vector): Vector =
    Vectors.fromBreeze(
      models
        .map(model => {
          val tmp = DenseVector.zeros[Double](numClasses)
          tmp(model.model.predict(features).ceil.toInt) = 1.0
          val res = model.weight * tmp
          res
        })
        .reduce(_ + _))

  override def copy(extra: ParamMap): BoostingClassificationModel = {
    val copied = new BoostingClassificationModel(uid, numClasses, models)
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

    private case class Data(error: Double, weight: Double)

    override protected def saveImpl(path: String): Unit = {
      val extraJson = "numClasses" -> instance.numClasses
      BoostingParams.saveImpl(path, instance, sc, Some(extraJson))
      instance.models.map(_.model.asInstanceOf[MLWritable]).zipWithIndex.foreach {
        case (model, idx) =>
          val modelPath = new Path(path, s"model-$idx").toString
          model.save(modelPath)
      }
      instance.models.zipWithIndex.foreach {
        case (model, idx) =>
          val data = Data(model.error, model.weight)
          val dataPath = new Path(path, s"data-$idx").toString
          sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
      }

    }
  }

  private class BoostingClassificationModelReader extends MLReader[BoostingClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BoostingClassificationModel].getName

    override def load(path: String): BoostingClassificationModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, _) = BoostingParams.loadImpl(path, sc, className)
      val numModels = metadata.getParamValue("maxIter").extract[Int]
      val models = (0 until numModels).toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[PredictionModel[Vector, _]](modelPath, sc)
      }
      val boostsData = (0 until numModels).toArray.map { idx =>
        val dataPath = new Path(path, s"data-$idx").toString
        val data = sparkSession.read.parquet(dataPath).select("error", "weight").head()
        (data.getAs[Double](0), data.getAs[Double](1))
      }
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val bcModel =
        new BoostingClassificationModel(metadata.uid, numClasses, boostsData.zip(models).map {
          case ((e, w), m) => new BoostedPredictionModel(e, w, m)
        })
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
