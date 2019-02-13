package org.apache.spark.ml.bagging

import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg.Vector

class PatchedPredictionModel(val indices: Array[Int], val model: PredictionModel[Vector, _]) extends Serializable {

}
