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
import org.apache.spark.ml.ensemble.{
  EnsemblePredictionModelType,
  EnsemblePredictorType,
  HasBaseLearners,
  HasStacker
}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.apache.spark.ml.stacking.StackingParams
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.ThreadUtils
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{parse, render}
import org.json4s.{DefaultFormats, JObject}

import scala.concurrent.Future
import scala.concurrent.duration.Duration

private[ml] trait StackingClassifierParams extends StackingParams with ClassifierParams {}

private[ml] object StackingClassifierParams {

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
    val learners = HasBaseLearners.loadImpl(path, sc)
    val stacker = HasStacker.loadImpl(path, sc)
    (metadata, learners, stacker)

  }

}

class StackingClassifier(override val uid: String)
    extends Predictor[Vector, StackingClassifier, StackingClassificationModel]
    with StackingClassifierParams
    with MLWritable {

  def setBaseLearners(value: Array[Predictor[_, _, _]]): this.type =
    set(baseLearners, value.map(_.asInstanceOf[EnsemblePredictorType]))

  def setStacker(value: Predictor[_, _, _]): this.type =
    set(stacker, value.asInstanceOf[EnsemblePredictorType])

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

      val handlePersistence = dataset.storageLevel == StorageLevel.NONE && (df.storageLevel == StorageLevel.NONE)
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

      val points = if (weightColIsUsed) {
        df.select($(labelCol), $(weightCol), $(featuresCol)).rdd.map {
          case Row(label: Double, weight: Double, features: Vector) =>
            Instance(label, weight, features)
        }
      } else {
        df.select($(labelCol), $(featuresCol)).rdd.map {
          case Row(label: Double, features: Vector) =>
            Instance(label, 1.0, features)
        }
      }

      val predictions =
        points.map(
          point =>
            Instance(
              point.label,
              point.weight,
              Vectors.dense(models.map(model => model.predict(point.features)))))

      val predictionsDF = spark.createDataFrame(predictions)

      val stack = fitBaseLearner(
        getStacker,
        "label",
        "features",
        getPredictionCol,
        optWeightColName.map(_ => "weight"))(predictionsDF)

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
      sr.setBaseLearners(learners.map(_.asInstanceOf[Predictor[Vector, _, _]]))
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

  def this(models: Array[EnsemblePredictionModelType], stack: EnsemblePredictionModelType) =
    this(Identifiable.randomUID("StackingClassificationModel"), models, stack)

  override def predict(features: Vector): Double = {
    stack.predict(Vectors.dense(models.map(_.predict(features))))
  }

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
      instance.models.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach {
        case (model, idx) =>
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
