package org.apache.spark.ml.regression

import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.shared.{HasMaxIter, HasParallelism}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}
import org.apache.spark.sql.Dataset

trait BaggingRegressorParams extends PredictorParams with HasMaxIter with HasParallelism {

  /**
    * param for the estimator to be stacked with bagging
    *
    * @group param
    */
  val baseLearner: Param[Predictor[_, _, _]] = new Param(this, "baseLearner", "base learner that will get stacked with bagging")

  /** @group getParam */
  def getBaseLearner: Predictor[_, _, _] = $(baseLearner)

  /**
    * param for whether samples are drawn with replacement
    *
    * @group param
    */
  val replacement: Param[Boolean] = new BooleanParam(this, "replacement", "whether samples are drawn with replacement")

  /** @group getParam */
  def getReplacement: Boolean = $(replacement)

  /**
    * param for whether samples are drawn with replacement
    *
    * @group param
    */
  val replacementFeatures: Param[Boolean] = new BooleanParam(this, "replacementFeautres", "whether features sampling are drawn with replacement")

  /** @group getParam */
  def getReplacementFeatures: Boolean = $(replacementFeatures)

}

class BaggingRegressor(override val uid: String) extends Predictor[Vector, BaggingRegressor, BaggingRegressionModel] with BaggingRegressorParams {

  def this()=this(Identifiable.randomUID("BaggingRegressor"))

  // Parameters from BaggingRegressorParams:

  /** @group setParam */
  def setBaseLearner(value: Predictor[_, _, _]): this.type = set(baseLearner, value)

  /** @group setParam */
  def setReplacement(value: Boolean): this.type = set(replacement, value)

  /** @group setParam */
  def setReplacementFeatures(value: Boolean): this.type = set(replacementFeatures, value)

  override def copy(extra: ParamMap): BaggingRegressor = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): BaggingRegressionModel = instrumented { instr =>

    null

  }
}

class BaggingRegressionModel(override val uid: String) extends PredictionModel[Vector, BaggingRegressionModel] {

  def this()=this(Identifiable.randomUID("BaggingRegressor"))

  override def predict(features: Vector): Double = ???

  override def copy(extra: ParamMap): BaggingRegressionModel = defaultCopy(extra)

}
