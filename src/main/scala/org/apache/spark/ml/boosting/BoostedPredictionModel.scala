package org.apache.spark.ml.boosting
import org.apache.spark.ml.ensemble.EnsemblePredictionModelType

class BoostedPredictionModel(
    val error: Double,
    val weight: Double,
    val model: EnsemblePredictionModelType)
    extends Serializable {}
