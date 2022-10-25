/*
 * Copyright 2019 Pierre Nodet
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.classification

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.CachedDiffFunction
import breeze.optimize.{LBFGSB => BreezeLBFGSB}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.boosting.GBMParams
import org.apache.spark.ml.boosting._
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.classification.ProbabilisticClassifier
import org.apache.spark.ml.dummy.DummyClassificationModel
import org.apache.spark.ml.dummy.DummyClassifier
import org.apache.spark.ml.ensemble.EnsembleClassificationModelType
import org.apache.spark.ml.ensemble.EnsemblePredictionModelType
import org.apache.spark.ml.ensemble.EnsembleRegressorType
import org.apache.spark.ml.ensemble.HasBaseLearner
import org.apache.spark.ml.ensemble.Utils
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.optim.loss.RDDLossFunction
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.ml.param.ParamValidators
import org.apache.spark.ml.param.shared.HasParallelism
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.util.PeriodicRDDCheckpointer
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.ThreadUtils
import org.json4s.DefaultFormats
import org.json4s.JObject
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.JsonMethods.render

import java.util.Locale
import scala.concurrent.Future
import scala.concurrent.duration.Duration

private[ml] trait GBMClassifierParams extends GBMParams with HasParallelism {

  /**
   * strategy for the init predictions, can be the class-prior or the uniform distribution.
   * (case-insensitive) Supported: "uniform", "prior". (default = prior)
   *
   * @group param
   */
  val initStrategy: Param[String] =
    new Param(
      this,
      "initStrategy",
      s"""strategy for the init predictions, can be a constant optimized for the minimized loss, zero, or the base learner learned on labels, (case-insensitive). Supported options: ${GBMClassifierParams.supportedInitStrategy
          .mkString(",")}""",
      ParamValidators.inArray(GBMClassifierParams.supportedInitStrategy))

  /** @group getParam */
  def getInitStrategy: String = $(initStrategy).toLowerCase(Locale.ROOT)

  /**
   * Loss function which GBM tries to minimize. (case-insensitive) Supported: "logloss",
   * "exponential", "binomial". (default = logloss)
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

  setDefault(loss -> "logloss")
  setDefault(initStrategy -> "prior")

}

private[ml] object GBMClassifierParams {

  final val supportedLossTypes: Array[String] =
    Array("logloss", "exponential", "binomial").map(_.toLowerCase(Locale.ROOT))

  final val supportedInitStrategy: Array[String] =
    Array("uniform", "prior").map(_.toLowerCase(Locale.ROOT))

  def loss(loss: String, numClasses: Int): GBMClassificationLoss =
    loss match {
      case "logloss" => LogLoss(numClasses)
      case "exponential" => ExponentialLoss
      case "binomial" => BinomialLoss
      case _ => throw new RuntimeException(s"GBMClassifier was given bad loss type: $loss")
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

  def loadImpl(
      path: String,
      sc: SparkContext,
      expectedClassName: String): (DefaultParamsReader.Metadata, EnsembleRegressorType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val learner = HasBaseLearner.loadImpl[EnsembleRegressorType](path, sc)
    (metadata, learner)
  }

}

class GBMClassifier(override val uid: String)
    extends ProbabilisticClassifier[Vector, GBMClassifier, GBMClassificationModel]
    with GBMClassifierParams
    with MLWritable {

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setBaseLearner(value: EnsembleRegressorType): this.type =
    set(baseLearner, value)

  /** @group setParam */
  def setNumBaseLearners(value: Int): this.type = set(numBaseLearners, value)

  /** @group setParam */
  def setReplacement(value: Boolean): this.type = set(replacement, value)

  /** @group setParam */
  def setSubsampleRatio(value: Double): this.type = set(subsampleRatio, value)

  /** @group setParam */
  def setSubspaceRatio(value: Double): this.type = set(subspaceRatio, value)

  /** @group setParam */
  def setInitStrategy(value: String): this.type = set(initStrategy, value)

  /** @group setParam */
  def setLoss(value: String): this.type = set(loss, value)

  /** @group setParam */
  def setLearningRate(value: Double): this.type = set(learningRate, value)

  /** @group expertSetParam */
  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  def setTol(value: Double): this.type = set(tol, value)

  /** @group expertSetParam */
  def setOptimizedWeights(value: Boolean): this.type = set(optimizedWeights, value)

  /** @group expertSetParam */
  def setUpdates(value: String): this.type = set(updates, value)

  /** @group setParam */
  def setValidationIndicatorCol(value: String): this.type = set(validationIndicatorCol, value)

  /** @group setParam */
  def setValidationTol(value: Double): this.type = set(validationTol, value)

  /** @group setParam */
  def setNumRounds(value: Int): this.type = set(numRounds, value)

  /** @group setParam */
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** @group setParam */
  def setParallelism(value: Int): this.type = set(parallelism, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

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
        initStrategy,
        loss,
        numBaseLearners,
        learningRate,
        optimizedWeights,
        validationIndicatorCol,
        subsampleRatio,
        replacement,
        subspaceRatio,
        maxIter,
        tol,
        seed)

      val spark = dataset.sparkSession
      val sc = spark.sparkContext

      val withValidation = isDefined(validationIndicatorCol) && $(validationIndicatorCol).nonEmpty

      val (trainDataset, validationDataset) = if (withValidation) {
        (
          extractInstances(dataset.filter(not(col($(validationIndicatorCol))))),
          extractInstances(dataset.filter(col($(validationIndicatorCol)))))
      } else {
        (extractInstances(dataset), null)
      }

      trainDataset
        .persist(StorageLevel.MEMORY_AND_DISK)
      if (withValidation) validationDataset.persist(StorageLevel.MEMORY_AND_DISK)

      val numInstances = trainDataset.count().toInt

      val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
      val numClasses = getNumClasses(dataset)
      instr.logNumClasses(numClasses)
      validateNumClasses(numClasses)

      val models = Array.ofDim[EnsemblePredictionModelType]($(numBaseLearners), numClasses)
      val subspaces =
        Array.tabulate($(numBaseLearners))(i =>
          subspace($(subspaceRatio), numFeatures, $(seed) + i))
      val weights = Array.ofDim[Double]($(numBaseLearners), numClasses)

      val gbmLoss = GBMClassifierParams.loss(getLoss, numClasses)
      val dim = gbmLoss.dim

      val init = if (getInitStrategy == "prior" && dim == 1 && numClasses == 2) {
        val probability = new DummyClassifier()
          .setStrategy("prior")
          .fit(spark.createDataFrame(trainDataset))
          .probability
        val logodds =
          Vectors.dense(math.log(probability(1) / (1 - probability(1))))
        new DummyClassificationModel(1, logodds, logodds)
          .setStrategy("constant")
      } else {
        new DummyClassifier()
          .setStrategy(getInitStrategy)
          .fit(spark.createDataFrame(trainDataset))
      }

      val lowerBounds = BDV.fill(dim)(0.0)
      val upperBounds = BDV.fill(dim)(Double.PositiveInfinity)
      val optimizer = new BreezeLBFGSB(lowerBounds, upperBounds, $(maxIter), 10, $(tol))

      var predictions = trainDataset.map(instance => {
        init.predictRaw(instance.features).copy.toArray
      })
      val predictionsCheckpointer = new PeriodicRDDCheckpointer[Array[Double]](
        $(checkpointInterval),
        sc,
        StorageLevel.MEMORY_AND_DISK)
      predictionsCheckpointer.update(predictions)

      var validationPredictions: RDD[Array[Double]] = null
      var validatePredictionsCheckpointer: PeriodicRDDCheckpointer[Array[Double]] = null
      var bestValidationError: Double = 0.0
      if (withValidation) {
        validationPredictions = validationDataset.map(instance => {
          init.predictRaw(instance.features).copy.toArray
        })
        validatePredictionsCheckpointer = new PeriodicRDDCheckpointer[Array[Double]](
          $(checkpointInterval),
          sc,
          StorageLevel.MEMORY_AND_DISK)
        validatePredictionsCheckpointer.update(validationPredictions)
        bestValidationError = validationDataset
          .zip(validationPredictions)
          .map { case (instance, prediction) =>
            gbmLoss.loss(gbmLoss.encodeLabel(instance.label), prediction.toArray)
          }
          .mean()
      }

      var i = 0
      var v = 0
      while (i < $(numBaseLearners) && v < $(numRounds)) {

        val subspace = subspaces(i)

        val bag = trainDataset
          .zip(predictions)
          .sample($(replacement), $(subsampleRatio), $(seed))

        val subbag = bag.map { case (instance, prediction) =>
          (instance.copy(features = slice(subspace)(instance.features)), prediction)
        }

        val pseudoResiduals = gbmLoss match {
          case gbmLoss: HasHessian if (getUpdates == "newton") =>
            val hessians: RDD[Array[Double]] = subbag.map { case (instance, prediction) =>
              gbmLoss
                .hessian(gbmLoss.encodeLabel(instance.label), prediction)
                .map(h => math.max(h, 1e-2))
            }
            val sumHessians =
              hessians.treeReduce(
                { case (acc, vec) =>
                  val res = Array.ofDim[Double](dim)
                  var i = 0
                  while (i < dim) {
                    res(i) = acc(i) + vec(i)
                    i += 1
                  }
                  res
                },
                $(aggregationDepth))
            subbag.zip(hessians).map { case ((instance, prediction), hessian) =>
              val negGrad =
                gbmLoss.negativeGradient(gbmLoss.encodeLabel(instance.label), prediction)
              val label = Array.ofDim[Double](dim)
              val weight = Array.ofDim[Double](dim)
              var i = 0
              while (i < dim) {
                label(i) = negGrad(i) / hessian(i)
                weight(i) = 1.0 / 2.0 * hessian(i) / sumHessians(i) * instance.weight
                i += 1
              }
              (label, weight, instance.features)
            }
          case gbmLoss =>
            subbag.map { case (instance, prediction) =>
              val negGrad =
                gbmLoss.negativeGradient(gbmLoss.encodeLabel(instance.label), prediction)
              (negGrad, Array.fill(dim)(instance.weight), instance.features)
            }
        }

        val futureWeightedModels = Array
          .range(0, dim)
          .map { j =>
            Future[EnsemblePredictionModelType] {

              val instances = pseudoResiduals.map { case (labels, weights, features) =>
                Instance(labels(j), weights(j), features)
              }
              instances.persist(StorageLevel.MEMORY_AND_DISK)

              val featuresMetadata =
                Utils.getFeaturesMetadata(dataset, $(featuresCol), Some(subspace))

              val df = spark
                .createDataFrame(instances)
                .withMetadata("features", featuresMetadata)

              val model =
                fitBaseLearner(
                  $(baseLearner),
                  "label",
                  "features",
                  $(predictionCol),
                  Some("weight"))(df)

              instances.unpersist()

              model

            }(getExecutionContext)

          }

        val imodels =
          futureWeightedModels.map(ThreadUtils.awaitResult(_, Duration.Inf))

        val solution = $(optimizedWeights) match {
          case true =>
            val instances = subbag.map { case (instance, prediction) =>
              GBMLossInstance(
                gbmLoss.encodeLabel(instance.label),
                instance.weight,
                prediction,
                imodels.map(_.predict(instance.features)))
            }
            instances.persist(StorageLevel.MEMORY_AND_DISK)
            val getAggregatorFunc =
              new GBMLossAggregator(gbmLoss)(_)
            val costFun =
              new RDDLossFunction(instances, getAggregatorFunc, None, $(aggregationDepth))
            val alpha = optimizer.minimize(new CachedDiffFunction(costFun), BDV.ones(dim))
            instances.unpersist()
            alpha.toArray
          case false => Array.fill(dim)(1.0)
        }
        val iweights = solution.map(_ * $(learningRate))

        models(i) = imodels
        weights(i) = iweights

        predictions = trainDataset
          .zip(predictions)
          .map { case (instance, prediction) =>
            val sliced = slice(subspace)(instance.features)
            val res = Array.ofDim[Double](dim)
            var j = 0
            while (j < dim) {
              res(j) = prediction(j) + iweights(j) * imodels(j).predict(sliced)
              j += 1
            }
            res
          }
        predictionsCheckpointer.update(predictions)

        if (withValidation) {
          validationPredictions = validationDataset
            .zip(validationPredictions)
            .map { case (instance, prediction) =>
              val sliced = slice(subspace)(instance.features)
              val res = Array.ofDim[Double](dim)
              var j = 0
              while (j < dim) {
                res(j) = prediction(j) + iweights(j) * imodels(j).predict(sliced)
                j += 1
              }
              res
            }
          validatePredictionsCheckpointer.update(validationPredictions)
          val validationError = validationDataset
            .zip(validationPredictions)
            .map { case (instance, prediction) =>
              gbmLoss.loss(gbmLoss.encodeLabel(instance.label), prediction)
            }
            .mean()
          if (bestValidationError - validationError < $(validationTol) * math.max(
              validationError,
              0.01)) {
            v += 1
          } else if (validationError < bestValidationError) {
            bestValidationError = validationError
            v = 0
          }
        }

        i += 1

      }

      trainDataset.unpersist()
      if (withValidation) validationDataset.unpersist()

      new GBMClassificationModel(
        numClasses,
        weights.take(i - v),
        subspaces.take(i - v),
        models.take(i - v),
        init,
        dim)

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
    val weights: Array[Array[Double]],
    val subspaces: Array[Array[Int]],
    val models: Array[Array[EnsemblePredictionModelType]],
    val init: EnsembleClassificationModelType,
    val dim: Int)
    extends ProbabilisticClassificationModel[Vector, GBMClassificationModel]
    with GBMClassifierParams
    with MLWritable {

  def this(
      numClasses: Int,
      weights: Array[Array[Double]],
      subspaces: Array[Array[Int]],
      models: Array[Array[EnsemblePredictionModelType]],
      init: EnsembleClassificationModelType,
      dim: Int) =
    this(
      Identifiable.randomUID("GBMClassificationModel"),
      numClasses,
      weights,
      subspaces,
      models,
      init,
      dim)

  val numModels: Int = models.length

  private def gbmLoss = GBMClassifierParams.loss(getLoss, numClasses)

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector =
    gbmLoss.raw2probabilityInPlace(rawPrediction)

  override def predictRaw(features: Vector): Vector = {

    val res = init.predictRaw(features).copy.toArray

    var i = 0
    while (i < numModels) {
      val sliced = slice(subspaces(i))(features)
      var j = 0
      while (j < dim) {
        res(j) += models(i)(j).predict(sliced) * weights(i)(j)
        j += 1
      }
      i += 1
    }

    // handle binary and multi class classification
    if (dim == 1 && numClasses == 2) {
      return Vectors.dense(-res(0), res(0))
    } else {
      return Vectors.dense(res)
    }

  }

  override def copy(extra: ParamMap): GBMClassificationModel = {
    val copied =
      new GBMClassificationModel(uid, numClasses, weights, subspaces, models, init, dim)
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

    private case class Data(weight: Double, subspace: Array[Int])

    override protected def saveImpl(path: String): Unit = {
      val extraJson =
        ("numClasses" -> instance.numClasses) ~ ("numModels" -> instance.numModels) ~ ("dim" -> instance.dim)
      GBMClassifierParams.saveImpl(instance, path, sc, Some(extraJson))
      val initPath = new Path(path, s"init").toString
      instance.init.asInstanceOf[MLWritable].save(initPath)
      instance.models.zipWithIndex.foreach { case (models, idx) =>
        models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach { case (model, k) =>
          val modelPath = new Path(path, s"model-$idx-$k").toString
          model.save(modelPath)
        }
      }
      instance.weights.zip(instance.subspaces).zipWithIndex.foreach {
        case ((weights, subspace), idx) =>
          weights.zipWithIndex.foreach { case (weight, k) =>
            val data = Data(weight, subspace)
            val dataPath = new Path(path, s"data-$idx-$k").toString
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
      val numModels = (metadata.metadata \ "numModels").extract[Int]
      val dim = (metadata.metadata \ "dim").extract[Int]
      val initPath = new Path(path, s"init").toString
      val init =
        DefaultParamsReader.loadParamsInstance[EnsembleClassificationModelType](initPath, sc)
      val models = (0 until numModels).toArray.map { idx =>
        (0 until dim).map { k =>
          val modelPath = new Path(path, s"model-$idx-$k").toString
          DefaultParamsReader
            .loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
        }.toArray
      }
      val boostsData = (0 until numModels).toArray.map { idx =>
        (0 until dim)
          .map { k =>
            val dataPath = new Path(path, s"data-$idx-$k").toString
            val data =
              sparkSession.read.json(dataPath).select("weight", "subspace").head()
            (data.getAs[Double](0), data.getAs[Seq[Long]](1).map(_.toInt).toArray)
          }
          .toArray
          .unzip
      }.unzip
      val bcModel =
        new GBMClassificationModel(
          metadata.uid,
          numClasses,
          boostsData._1,
          boostsData._2.map(_(0)),
          models,
          init,
          dim)
      metadata.getAndSetParams(bcModel)
      bcModel
    }
  }
}
