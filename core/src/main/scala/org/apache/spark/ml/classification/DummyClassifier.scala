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

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.classification.ProbabilisticClassifier
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.ParamValidators
import org.apache.spark.ml.param.shared.HasTol
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql.Dataset
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._

import java.util.Locale

trait DummyClassifierParams extends PredictorParams with HasTol {

  /**
   * strategy to use to generate predictions. (case-insensitive) Supported: "uniform", "prior",
   * "constant". (default = uniform)
   *
   * @group param
   */
  val strategy: Param[String] =
    new Param(
      this,
      "init",
      s"""strategy to use to generate predictions, (case-insensitive). Supported options: ${DummyClassifierParams.supportedStrategy
          .mkString(",")}""",
      ParamValidators.inArray(DummyClassifierParams.supportedStrategy))

  def getStrategy: String = $(strategy).toLowerCase(Locale.ROOT)

  /**
   * param for the constant predicted by the predictor
   *
   * @group param
   */
  val constant: Param[Double] =
    new DoubleParam(this, "constant", "constant predicted by the predictor")

  /** @group getParam */
  def getConstant: Double = $(constant)

  setDefault(strategy, "uniform")

}

object DummyClassifierParams {
  final val supportedStrategy: Array[String] =
    Array("uniform", "prior", "constant").map(_.toLowerCase(Locale.ROOT))
}

class DummyClassifier(override val uid: String)
    extends ProbabilisticClassifier[Vector, DummyClassifier, DummyClassificationModel]
    with DummyClassifierParams
    with DefaultParamsWritable {

  override def copy(extra: ParamMap): DummyClassifier = defaultCopy(extra)

  def this() = this(Identifiable.randomUID("DummyClassifier"))

  /** @group setParam */
  def setStrategy(value: String): this.type =
    set(strategy, value)

  /** @group setParam */
  def setConstant(value: Double): this.type =
    set(constant, value)

  override protected def train(dataset: Dataset[_]): DummyClassificationModel = instrumented {
    instr =>
      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(this, labelCol, featuresCol, predictionCol, strategy, constant)

      val numClasses = getNumClasses(dataset)
      instr.logNumClasses(numClasses)
      validateNumClasses(numClasses)

      val (rawPrediction, probability) = getStrategy match {
        case "uniform" => (Array.fill(numClasses)(0.0), Array.fill(numClasses)(1.0 / numClasses))
        case "prior" => {
          val numInstances = dataset.count().toDouble
          val priors = dataset.groupBy("label").count().collect().map { row =>
            (row.getDouble(0).toInt, row.getLong(1))
          }
          val sorted = priors.sortBy(_._1).map(_._2 / numInstances)
          (sorted.map(math.log(_)), sorted)
        }
        case "constant" => {
          validateLabel($(constant), numClasses)
          val tmp = Array.fill(numClasses)(0.0)
          tmp($(constant).toInt) = 1.0
          (tmp, tmp)
        }
      }

      new DummyClassificationModel(
        numClasses,
        Vectors.dense(rawPrediction),
        Vectors.dense(probability))

  }

}

object DummyClassifier extends DefaultParamsReadable[DummyClassifier]

class DummyClassificationModel(
    override val uid: String,
    override val numClasses: Int,
    val rawPrediction: Vector,
    val probability: Vector)
    extends ProbabilisticClassificationModel[Vector, DummyClassificationModel]
    with DummyClassifierParams
    with MLWritable {

  def this(numClasses: Int, rawPrediction: Vector, probability: Vector) =
    this(
      Identifiable.randomUID("DummyClassificationModel"),
      numClasses,
      rawPrediction,
      probability)

  override def copy(extra: ParamMap): DummyClassificationModel = {
    val copied =
      new DummyClassificationModel(uid, numClasses, rawPrediction, probability).setParent(parent)
    copyValues(copied, extra)
  }

  /** @group setParam */
  def setStrategy(value: String): this.type =
    set(strategy, value)

  /** @group setParam */
  def setConstant(value: Double): this.type =
    set(constant, value)

  /** @group setParam */
  def setTol(value: Double): this.type =
    set(tol, value)

  override def predictRaw(features: Vector): Vector = rawPrediction

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = probability

  override def write: MLWriter = new DummyClassificationModel.DummyClassificationModelWriter(this)

  override def toString: String = {
    s"DummyClassificationModel: uid=$uid, rawPrediction=$rawPrediction, probability=$probability"
  }

}

object DummyClassificationModel extends MLReadable[DummyClassificationModel] {

  override def read: MLReader[DummyClassificationModel] =
    new DummyClassificationModelReader

  override def load(path: String): DummyClassificationModel = super.load(path)

  private[DummyClassificationModel] class DummyClassificationModelWriter(
      instance: DummyClassificationModel)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(
        instance,
        path,
        sc,
        Some(
          ("rawPrediction" -> instance.rawPrediction.toArray.toList) ~
            ("probability" -> instance.probability.toArray.toList) ~
            ("numClasses" -> instance.numClasses)))
    }
  }

  private class DummyClassificationModelReader extends MLReader[DummyClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[DummyClassificationModel].getName

    override def load(path: String): DummyClassificationModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val rawPrediction = (metadata.metadata \ "rawPrediction").extract[List[Double]]
      val probability = (metadata.metadata \ "probability").extract[List[Double]]
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val model =
        new DummyClassificationModel(
          metadata.uid,
          numClasses,
          Vectors.dense(rawPrediction.toArray),
          Vectors.dense(probability.toArray))
      metadata.getAndSetParams(model)
      model
    }
  }

}
