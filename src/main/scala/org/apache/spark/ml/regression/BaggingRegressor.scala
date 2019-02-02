package org.apache.spark.ml.regression

import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasMaxIter
import org.apache.spark.sql.Dataset

trait BaggingRegressorParams extends PredictorParams with HasMaxIter {}

class BaggingRegressor[T <: Predictor[Vector, _, _]](override val uid: String) extends Predictor[Vector, BaggingRegressor[T], BaggingRegressionModel[T]] {

  override def copy(extra: ParamMap): BaggingRegressor[T] = ???

  override protected def train(dataset: Dataset[_]): BaggingRegressionModel[T] = ???
}

class BaggingRegressionModel[T <: Predictor[_, _, _]] extends PredictionModel[Vector, BaggingRegressionModel[T]] {

  override def predict(features: Vector): Double = ???

  override def copy(extra: ParamMap): BaggingRegressionModel[T] = ???

  override val uid: String = ""
}
