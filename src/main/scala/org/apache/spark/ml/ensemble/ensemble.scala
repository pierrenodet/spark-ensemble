package org.apache.spark.ml
import org.apache.spark.ml.linalg.Vector

package object ensemble {

  type EnsemblePredictorType = Predictor[Vector, E, M] forSome {
    type M <: PredictionModel[Vector, M]
    type E <: Predictor[Vector, E, M]
  }

  type EnsemblePredictionModelType = PredictionModel[Vector, M] forSome {
    type M <: PredictionModel[Vector, M]
  }

}
