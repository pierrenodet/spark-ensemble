package org.apache.spark.ml.param
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.linalg.Vector

trait PredictorVectorTypeTrait {

  type PredictorVectorType = Predictor[Vector, E, M] forSome {
    type M <: PredictionModel[Vector, M]
    type E <: Predictor[Vector, E, M]
  }

}
