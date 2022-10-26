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

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.ensemble.EnsemblePredictionModelType
import org.apache.spark.ml.ensemble.EnsemblePredictorType
import org.apache.spark.ml.ensemble.HasBaseLearners
import org.apache.spark.ml.ensemble.HasStacker
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.stacking.StackingParams
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql.Dataset
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

private[ml] trait StackingClassifierParams
    extends StackingParams[EnsemblePredictorType]
    with ClassifierParams {

  /**
   * Discrete (SAMME) or Real (SAMME.R) boosting algorithm. (case-insensitive) Supported: "class",
   * "raw", "proba". (default = class)
   *
   * @group param
   */
  val stackMethod: Param[String] =
    new Param(
      this,
      "stackMethod",
      "stackMethod, (case-insensitive). Supported options:" + s"${StackingClassifierParams.supportedStackMethod
          .mkString(",")}",
      (value: String) =>
        StackingClassifierParams.supportedStackMethod.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getStackMethod: String = $(stackMethod).toLowerCase(Locale.ROOT)

  setDefault(stackMethod -> "class")

}

private[ml] object StackingClassifierParams {

  final val supportedStackMethod: Array[String] =
    Array("class", "raw", "proba").map(_.toLowerCase(Locale.ROOT))

  def saveImpl(
      instance: StackingClassifierParams,
      path: String,
      sc: SparkContext,
      extraMetadata: Option[JObject] = None): Unit = {

    val params = instance.extractParamMap().toSeq
    val jsonParams = render(
      params
        .filter { case ParamPair(p, _) => p.name != "baseLearners" && p.name != "stacker" }
        .map { case ParamPair(p, v) => p.name -> parse(p.jsonEncode(v)) }
        .toList)

    DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, Some(jsonParams))
    HasBaseLearners.saveImpl(instance, path, sc)
    HasStacker.saveImpl(instance, path, sc)

  }

  def loadImpl(path: String, sc: SparkContext, expectedClassName: String)
      : (DefaultParamsReader.Metadata, Array[EnsemblePredictorType], EnsemblePredictorType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val learners = HasBaseLearners.loadImpl[EnsemblePredictorType](path, sc)
    val stacker = HasStacker.loadImpl[EnsemblePredictorType](path, sc)
    (metadata, learners, stacker)

  }

}

class StackingClassifier(override val uid: String)
    extends Predictor[Vector, StackingClassifier, StackingClassificationModel]
    with StackingClassifierParams
    with MLWritable {

  def setBaseLearners(value: Array[EnsemblePredictorType]): this.type =
    set(baseLearners, value)

  def setStacker(value: EnsemblePredictorType): this.type =
    set(stacker, value)

  def setStackMethod(value: String): this.type =
    set(stackMethod, value)

  def setParallelism(value: Int): this.type = set(parallelism, value)

  def this() = this(Identifiable.randomUID("StackingClassifier"))

  override def copy(extra: ParamMap): StackingClassifier = {
    val copied = new StackingClassifier(uid)
    copyValues(copied, extra)
    copied.setBaseLearners(copied.getBaseLearners.map(_.copy(extra)))
    copied.setStacker(copied.getStacker.copy(extra))
  }

  override protected def train(dataset: Dataset[_]): StackingClassificationModel = instrumented {
    instr =>
      val spark = dataset.sparkSession

      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(this, labelCol, weightCol, featuresCol, predictionCol, parallelism)

      val weightColIsUsed = isDefined(weightCol) && $(weightCol).nonEmpty && {
        getBaseLearners.forall {
          case _: HasWeightCol => true
          case c =>
            instr.logWarning(s"weightCol is ignored, as it is not supported by $c now.")
            false
        }
      }

      val df = if (weightColIsUsed) {
        dataset.select($(labelCol), $(weightCol), $(featuresCol))
      } else {
        dataset.select($(labelCol), $(featuresCol))
      }

      val optWeightColName = if (weightColIsUsed) {
        Some($(weightCol))
      } else {
        None
      }

      val handlePersistence =
        dataset.storageLevel == StorageLevel.NONE && (df.storageLevel == StorageLevel.NONE)
      if (handlePersistence) {
        df.persist(StorageLevel.MEMORY_AND_DISK)
      }

      val numBaseLearners = getBaseLearners.length

      val futureModels = (0 until numBaseLearners).map(iter =>
        Future[EnsemblePredictionModelType] {

          fitBaseLearner(
            getBaseLearners(iter),
            getLabelCol,
            getFeaturesCol,
            getPredictionCol,
            optWeightColName)(df)

        }(getExecutionContext))

      val models = futureModels.map(f => ThreadUtils.awaitResult(f, Duration.Inf)).toArray

      val instances = extractInstances(df)

      val predictions =
        instances.map(instance => {
          val features = models.flatMap(model =>
            model match {
              case model: ProbabilisticClassificationModel[Vector, _]
                  if (getStackMethod == "proba") =>
                model.predictProbability(instance.features).toArray
              case model: ClassificationModel[Vector, _] if (getStackMethod == "raw") =>
                model.predictRaw(instance.features).toArray
              case _ => Array(model.predict(instance.features))
            })
          Instance(instance.label, instance.weight, Vectors.dense(features))
        })

      val predictionsDF = spark
        .createDataFrame(predictions)

      val stack =
        fitBaseLearner($(stacker), "label", "features", getPredictionCol, Some("weight"))(
          predictionsDF)

      df.unpersist()

      new StackingClassificationModel(models, stack)

  }

  override def write: MLWriter = new StackingClassifier.StackingClassifierWriter(this)

}

object StackingClassifier extends MLReadable[StackingClassifier] {

  override def read: MLReader[StackingClassifier] = new StackingClassifierReader

  override def load(path: String): StackingClassifier = super.load(path)

  private[StackingClassifier] class StackingClassifierWriter(instance: StackingClassifier)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      StackingClassifierParams.saveImpl(instance, path, sc)
    }

  }

  private class StackingClassifierReader extends MLReader[StackingClassifier] {

    /** Checked against metadata when loading model */
    private val className = classOf[StackingClassifier].getName

    override def load(path: String): StackingClassifier = {
      val (metadata, learners, stacker) = StackingClassifierParams.loadImpl(path, sc, className)
      val sr = new StackingClassifier(metadata.uid)
      metadata.getAndSetParams(sr)
      sr.setBaseLearners(learners)
      sr.setStacker(stacker)
    }
  }

}

class StackingClassificationModel(
    override val uid: String,
    val models: Array[EnsemblePredictionModelType],
    val stack: EnsemblePredictionModelType)
    extends PredictionModel[Vector, StackingClassificationModel]
    with StackingClassifierParams
    with MLWritable {

  override def predict(features: Vector): Double = {
    val predictions = models.flatMap(model =>
      model match {
        case model: ProbabilisticClassificationModel[Vector, _] if (getStackMethod == "proba") =>
          model.predictProbability(features).toArray
        case model: ClassificationModel[Vector, _] if (getStackMethod == "raw") =>
          model.predictRaw(features).toArray
        case _ => Array(model.predict(features))
      })
    stack.predict(Vectors.dense(predictions))
  }

  def this(models: Array[EnsemblePredictionModelType], stack: EnsemblePredictionModelType) =
    this(Identifiable.randomUID("StackingClassificationModel"), models, stack)

  override def copy(extra: ParamMap): StackingClassificationModel = {
    val copied = new StackingClassificationModel(uid, models, stack)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter =
    new StackingClassificationModel.StackingClassificationModelWriter(this)

}

object StackingClassificationModel extends MLReadable[StackingClassificationModel] {

  override def read: MLReader[StackingClassificationModel] = new StackingClassificationModelReader

  override def load(path: String): StackingClassificationModel = super.load(path)

  private[StackingClassificationModel] class StackingClassificationModelWriter(
      instance: StackingClassificationModel)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      StackingClassifierParams.saveImpl(instance, path, sc)
      instance.models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach { case (model, idx) =>
        val modelPath = new Path(path, s"model-$idx").toString
        model.save(modelPath)
      }
      val stackPath = new Path(path, "stack").toString
      instance.stack.asInstanceOf[MLWritable].save(stackPath)

    }
  }

  private class StackingClassificationModelReader extends MLReader[StackingClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[StackingClassificationModel].getName

    override def load(path: String): StackingClassificationModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val (metadata, baseLearners, stacker) =
        StackingClassifierParams.loadImpl(path, sc, className)
      val models = baseLearners.indices.toArray.map { idx =>
        val modelPath = new Path(path, s"model-$idx").toString
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](modelPath, sc)
      }
      val stackPath = new Path(path, "stack").toString
      val stack =
        DefaultParamsReader.loadParamsInstance[EnsemblePredictionModelType](stackPath, sc)
      val scModel = new StackingClassificationModel(metadata.uid, models, stack)
      metadata.getAndSetParams(scModel)
      scModel.set("baseLearners", baseLearners)
      scModel.set("stacker", stacker)
      scModel
    }
  }
}
