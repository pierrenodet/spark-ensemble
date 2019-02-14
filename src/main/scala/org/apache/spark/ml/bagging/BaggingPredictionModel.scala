package org.apache.spark.ml.bagging

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.util.ThreadUtils

import scala.concurrent.{ExecutionContext, Future}
import scala.concurrent.duration.Duration

trait BaggingPredictionModel {

  def predictNormal(features: Vector, models: Array[PatchedPredictionModel]): Array[Double] = {
    models.map(model => {
      val indices = model.indices
      val subFeatures = features match {
        case features: DenseVector => Vectors.dense(indices.map(features.apply))
        case features: SparseVector => features.slice(indices)
      }
      model.model.predict(subFeatures)
    })
  }

  def predictFuture(
      features: Vector,
      models: Array[PatchedPredictionModel],
      executionContext: ExecutionContext): Array[Double] = {
    val futurePredictions = models.map(model =>
      Future[Double] {
        val indices = model.indices
        val subFeatures = features match {
          case features: DenseVector => Vectors.dense(indices.map(features.apply))
          case features: SparseVector => features.slice(indices)
        }
        model.model.predict(subFeatures)
      }(executionContext))
    futurePredictions.map(ThreadUtils.awaitResult(_, Duration.Inf))
  }

}
