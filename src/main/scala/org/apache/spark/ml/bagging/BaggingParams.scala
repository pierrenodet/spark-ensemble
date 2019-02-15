package org.apache.spark.ml.bagging

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.ensemble.{HasBaseLearner, SubSpaceParams}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasParallelism}

trait BaggingParams
    extends PredictorParams
    with HasMaxIter
    with HasParallelism
    with HasBaseLearner
    with SubSpaceParams {}
