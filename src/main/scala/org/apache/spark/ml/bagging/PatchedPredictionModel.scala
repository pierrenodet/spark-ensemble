package org.apache.spark.ml.bagging

import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg.Vector

class PatchedPredictionModel(indices: Array[Int], model: PredictionModel[Vector, _]) extends Serializable {
  def getModel: PredictionModel[Vector, _] = model

  def getIndices: Array[Int] = indices
}