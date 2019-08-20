package org.apache.spark.ml.bagging

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.ensemble.{HasBaseLearner, HasNumBaseLearners, HasSubBag}
import org.apache.spark.ml.param.shared.{HasParallelism, HasWeightCol}

trait BaggingParams
    extends PredictorParams
    with HasNumBaseLearners
    with HasParallelism
    with HasWeightCol
    with HasBaseLearner
    with HasSubBag {

  setDefault(numBaseLearners -> 10)
}
