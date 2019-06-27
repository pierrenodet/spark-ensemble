package org.apache.spark.ml.boosting
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.ensemble.{HasBaseLearner, HasLearningRate}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasSeed, HasWeightCol}

trait BoostingParams
    extends PredictorParams
    with HasMaxIter
    with HasWeightCol
    with HasSeed
    with HasBaseLearner
    with HasLearningRate {

  setDefault(maxIter -> 10)
        
}
