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

package org.apache.spark.ml

import org.apache.spark.ml.classification.{
  ClassificationModel,
  Classifier,
  ProbabilisticClassificationModel,
  ProbabilisticClassifier
}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.{RegressionModel, Regressor}

import scala.language.existentials

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

  type EnsembleProbabilisticClassificationModelType =
    ProbabilisticClassificationModel[Vector, M] forSome {
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
