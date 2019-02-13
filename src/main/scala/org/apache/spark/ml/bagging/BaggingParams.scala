package org.apache.spark.ml.bagging

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasMaxIter, HasParallelism}
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter, MLWritable}
import org.json4s.JObject
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

trait BaggingParams extends PredictorParams with HasMaxIter with HasParallelism with PredictorVectorTypeTrait {

  /**
    * param for the estimator to be stacked with bagging
    *
    * @group param
    */
  val baseLearner: Param[PredictorVectorType] =
    new Param[PredictorVectorType](
      this,
      "baseLearner",
      "base learner that will get stacked with bagging"
    )

  /** @group getParam */
  def getBaseLearner: PredictorVectorType = $(baseLearner)

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
  val replacementFeatures: Param[Boolean] =
    new BooleanParam(this, "replacementFeautres", "whether features sampling are drawn with replacement")

  /** @group getParam */
  def getReplacementFeatures: Boolean = $(replacementFeatures)

  setDefault(replacementFeatures -> false)

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val sampleRatioFeatures: Param[Double] =
    new DoubleParam(this, "sampleFeaturesNumber", "max number of features sampled out of the dataset")

  /** @group getParam */
  def getSampleRatioFeatures: Double = $(sampleRatioFeatures)

  setDefault(sampleRatioFeatures -> 1)

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val seed: Param[Long] = new LongParam(this, "seed", "seed for randomness")

  /** @group getParam */
  def getSeed: Long = $(seed)

  setDefault(seed -> System.nanoTime())

  setDefault(maxIter     -> 10)
  setDefault(parallelism -> 1)

}

object BaggingParams extends PredictorVectorTypeTrait {

  def saveImpl(
    path: String,
    instance: BaggingParams,
    sc: SparkContext,
    extraMetadata: Option[JObject] = None
  ): Unit = {

    val params = instance.extractParamMap().toSeq
    val jsonParams = render(
      params
        .filter { case ParamPair(p, v) => p.name != "baseLearner" }
        .map { case ParamPair(p, v) => p.name -> parse(p.jsonEncode(v)) }
        .toList
    )

    DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, Some(jsonParams))

    val learnerPath = new Path(path, "learner").toString
    instance.getBaseLearner.asInstanceOf[MLWritable].save(learnerPath)

  }

  def loadImpl(
    path: String,
    sc: SparkContext,
    expectedClassName: String
  ): (DefaultParamsReader.Metadata, PredictorVectorType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val learnerPath = new Path(path, "learner").toString
    val learner = DefaultParamsReader.loadParamsInstance[PredictorVectorType](learnerPath, sc)
    (metadata, learner)
  }

}
