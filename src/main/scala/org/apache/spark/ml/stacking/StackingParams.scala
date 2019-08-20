package org.apache.spark.ml.stacking
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.ensemble.{HasBaseLearners, HasStacker}
import org.apache.spark.ml.param.shared.{HasParallelism, HasWeightCol}

trait StackingParams
    extends PredictorParams
    with HasParallelism
    with HasWeightCol
    with HasStacker
    with HasBaseLearners {}
