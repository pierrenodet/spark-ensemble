package org.apache.spark.ml.bagging

import org.apache.spark.ml.ensemble.EnsemblePredictionModelType

class PatchedPredictionModel(val indices: Array[Int], val model: EnsemblePredictionModelType)
    extends Serializable {}
