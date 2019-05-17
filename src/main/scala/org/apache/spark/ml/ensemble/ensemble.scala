package org.apache.spark.ml

import org.apache.spark.ml.classification.{ClassificationModel, Classifier, ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.{RegressionModel, Regressor}

package object ensemble {

  type EnsemblePredictorType = Predictor[Vector, E, M] forSome {
    type M <: PredictionModel[Vector, M]
    type E <: Predictor[Vector, E, M]
  }

  type EnsemblePredictionModelType = PredictionModel[Vector, M] forSome {
    type M <: PredictionModel[Vector, M]
  }

  type EnsembleClassifierType = Classifier[Vector, E, M] forSome {
    type M <: ClassificationModel[Vector, M]
    type E <: Classifier[Vector, E, M]
  }

  type EnsembleClassificationModelType = ClassificationModel[Vector, M] forSome {
    type M <: ClassificationModel[Vector, M]
  }

  type EnsembleProbabilisticClassifierType = ProbabilisticClassifier[Vector, E, M] forSome {
    type M <: ProbabilisticClassificationModel[Vector, M]
    type E <: ProbabilisticClassifier[Vector, E, M]
  }

  type EnsembleProbabilisticClassificationModelType = ProbabilisticClassificationModel[Vector, M] forSome {
    type M <: ProbabilisticClassificationModel[Vector, M]
  }

  type EnsembleRegressorType = Regressor[Vector, E, M] forSome {
    type M <: RegressionModel[Vector, M]
    type E <: Regressor[Vector, E, M]
  }

  type EnsembleRegressionModelType = RegressionModel[Vector, M] forSome {
    type M <: RegressionModel[Vector, M]
  }

}
