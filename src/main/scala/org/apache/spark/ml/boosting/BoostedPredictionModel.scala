package org.apache.spark.ml.boosting
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg.Vector

class BoostedPredictionModel(error: Double, weight: Double, model: PredictionModel[Vector, _]) extends Serializable {
  def getModel: PredictionModel[Vector, _] = model

  def getError: Double = error

  def getWeight: Double = weight
}