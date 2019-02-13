package org.apache.spark.ml.boosting
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg.Vector

class BoostedPredictionModel(val error: Double, val weight: Double, val model: PredictionModel[Vector, _]) extends Serializable {

}
