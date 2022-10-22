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

package org.apache.spark.ml.dummy

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.ParamValidators
import org.apache.spark.ml.param.shared.HasTol
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.regression.Regressor
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._

import java.util.Locale

trait DummyRegressorParams extends PredictorParams with HasTol {

  /**
   * strategy to use to generate predictions. (case-insensitive) Supported: "mean", "median",
   * "quantile", "constant". (default = mean)
   *
   * @group param
   */
  val strategy: Param[String] =
    new Param(
      this,
      "init",
      s"""strategy to use to generate predictions, (case-insensitive). Supported options: ${DummyRegressorParams.supportedStrategy
          .mkString(",")}""",
      ParamValidators.inArray(DummyRegressorParams.supportedStrategy))

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

  /**
   * param for the quantile estimated predicted by the predictor when strategy='quantile'
   *
   * @group param
   */
  val quantile: Param[Double] =
    new DoubleParam(
      this,
      "quantile",
      "quantile estimated predicted by the predictor when strategy='quantile'")

  /** @group getParam */
  def getQuantile: Double = $(quantile)

  setDefault(strategy, "mean")
  setDefault(tol, 1e-2)

}

object DummyRegressorParams {
  final val supportedStrategy: Array[String] =
    Array("mean", "median", "quantile", "constant").map(_.toLowerCase(Locale.ROOT))
}

class DummyRegressor(override val uid: String)
    extends Regressor[Vector, DummyRegressor, DummyRegressionModel]
    with DummyRegressorParams
    with DefaultParamsWritable {

  override def copy(extra: ParamMap): DummyRegressor = defaultCopy(extra)

  def this() = this(Identifiable.randomUID("DummyRegressor"))

  /** @group setParam */
  def setStrategy(value: String): this.type =
    set(strategy, value)

  /** @group setParam */
  def setConstant(value: Double): this.type =
    set(constant, value)

  /** @group setParam */
  def setQuantile(value: Double): this.type =
    set(quantile, value)

  /** @group setParam */
  def setTol(value: Double): this.type =
    set(tol, value)

  override protected def train(dataset: Dataset[_]): DummyRegressionModel = instrumented {
    instr =>
      instr.logPipelineStage(this)
      instr.logDataset(dataset)
      instr.logParams(this, labelCol, featuresCol, predictionCol, strategy, constant, quantile)

      val prediction = getStrategy match {
        case "mean" => dataset.select(mean($(labelCol))).first().getDouble(0)
        case "median" | "quantile" =>
          val q = if (getStrategy == "median") 0.5 else $(quantile)
          dataset.stat.approxQuantile("label", Array(q), $(tol))(0)
        case "constant" => $(constant)
      }

      new DummyRegressionModel(prediction)

  }

}

object DummyRegressor extends DefaultParamsReadable[DummyRegressor]

class DummyRegressionModel(override val uid: String, val prediction: Double)
    extends RegressionModel[Vector, DummyRegressionModel]
    with DummyRegressorParams
    with MLWritable {

  /** @group setParam */
  def setStrategy(value: String): this.type =
    set(strategy, value)

  /** @group setParam */
  def setConstant(value: Double): this.type =
    set(constant, value)

  /** @group setParam */
  def setQuantile(value: Double): this.type =
    set(quantile, value)

  /** @group setParam */
  def setTol(value: Double): this.type =
    set(tol, value)

  override def write: MLWriter = new DummyRegressionModel.DummyRegressionModelWriter(this)

  override def copy(extra: ParamMap): DummyRegressionModel = {
    val copied = new DummyRegressionModel(uid, prediction).setParent(parent)
    copyValues(copied, extra)
  }

  def this(prediction: Double) =
    this(Identifiable.randomUID("DummyRegressionModel"), prediction)

  def predict(features: Vector): Double = prediction

  override def toString: String = {
    s"DummyRegressionModel: uid=$uid, prediction=$prediction"
  }

}

object DummyRegressionModel extends MLReadable[DummyRegressionModel] {

  override def read: MLReader[DummyRegressionModel] =
    new DummyRegressionModelReader

  override def load(path: String): DummyRegressionModel = super.load(path)

  private[DummyRegressionModel] class DummyRegressionModelWriter(instance: DummyRegressionModel)
      extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(
        instance,
        path,
        sc,
        Some("prediction" -> instance.prediction))
    }
  }

  private class DummyRegressionModelReader extends MLReader[DummyRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[DummyRegressionModel].getName

    override def load(path: String): DummyRegressionModel = {
      implicit val format: DefaultFormats = DefaultFormats
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val prediction = (metadata.metadata \ "prediction").extract[Double]
      val model =
        new DummyRegressionModel(metadata.uid, prediction)
      metadata.getAndSetParams(model)
      model
    }
  }

}
