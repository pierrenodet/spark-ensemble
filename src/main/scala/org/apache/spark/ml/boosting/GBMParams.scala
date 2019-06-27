package org.apache.spark.ml.boosting

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.ensemble.{HasBaseLearner, HasLearningRate}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasSeed, HasTol, HasWeightCol}

trait GBMParams
    extends PredictorParams
    with HasMaxIter
    with HasWeightCol
    with HasSeed
    with HasBaseLearner
    with HasLearningRate
    with HasTol {

  setDefault(tol -> 1E-3)
  setDefault(maxIter -> 10)
}
