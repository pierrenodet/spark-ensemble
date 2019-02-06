package org.apache.spark.ml.bagging

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.shared.{HasMaxIter, HasParallelism}
import org.apache.spark.ml.param._
import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}

trait BaggingParams extends PredictorParams with HasMaxIter with HasParallelism {

  /**
    * param for the estimator to be stacked with bagging
    *
    * @group param
    */
  val baseLearner: Param[Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]]] = new Param[Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]]](this, "baseLearner", "base learner that will get stacked with bagging")

  /** @group getParam */
  def getBaseLearner: Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]] = $(baseLearner)
  /**
    * param for whether samples are drawn with replacement
    *
    * @group param
    */
  val replacement: Param[Boolean] = new BooleanParam(this, "replacement", "whether samples are drawn with replacement")

  /** @group getParam */
  def getReplacement: Boolean = $(replacement)

  setDefault(replacement -> false)

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val sampleRatio: Param[Double] = new DoubleParam(this, "sampleRatio", "ratio of rows sampled out of the dataset")

  /** @group getParam */
  def getSampleRatio: Double = $(sampleRatio)

  setDefault(sampleRatio -> 1)

  /**
    * param for whether samples are drawn with replacement
    *
    * @group param
    */
  val replacementFeatures: Param[Boolean] = new BooleanParam(this, "replacementFeautres", "whether features sampling are drawn with replacement")

  /** @group getParam */
  def getReplacementFeatures: Boolean = $(replacementFeatures)

  setDefault(replacementFeatures -> false)

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val sampleFeaturesNumber: Param[Int] = new IntParam(this, "sampleFeaturesNumber", "max number of features sampled out of the dataset")

  /** @group getParam */
  def getSampleFeaturesNumber: Int = $(sampleFeaturesNumber)

  setDefault(sampleFeaturesNumber -> 10)

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val seed: Param[Long] = new LongParam(this, "seed", "seed for randomness")

  /** @group getParam */
  def getSeed: Long = $(seed)

  setDefault(seed -> System.nanoTime())

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val reduce: Param[Array[Double] => Double] = new Param(this, "reduce", "function to reduce the predictions of the models")

  /** @group getParam */
  def getReduce: Array[Double] => Double = $(reduce)

  setDefault(reduce -> { predictions: Array[Double] => predictions.sum / predictions.length })

  setDefault(maxIter -> 10)
  setDefault(parallelism -> 1)

}
