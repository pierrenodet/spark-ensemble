package org.apache.spark.ml.boosting

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.exp
import org.apache.spark.ml.ensemble.EnsemblePredictionModelType
import org.apache.spark.ml.linalg.Vector

object GBM {

  def kpredict(weights: Array[Array[Double]], models: Array[Array[EnsemblePredictionModelType]])(
      features: Vector): Array[Double] = {
    val kprediction = weights.zip(models).map {
      case (weightByK, modelByK) =>
        weightByK
          .zip(modelByK)
          .map { case (weight, model) => weight * model.predict(features) }
          .sum
    }
    (exp(DenseVector[Double](kprediction)) / sum(exp(DenseVector[Double](kprediction)))).toArray
  }

}
