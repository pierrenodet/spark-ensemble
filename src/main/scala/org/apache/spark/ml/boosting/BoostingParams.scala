package org.apache.spark.ml.boosting
import java.util.Locale

import org.apache.commons.math3.util.FastMath
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.bagging.BaggingParams.PredictorVectorType
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasMaxIter, HasWeightCol}
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter, MLWritable}
import org.json4s.JObject
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

trait BoostingParams
    extends PredictorParams
    with HasMaxIter
    with HasWeightCol
    with PredictorVectorTypeTrait {

  /**
   * param for the base learner to be stacked with boosting
   *
   * @group param
   */
  val baseLearner: Param[PredictorVectorType] =
    new Param[PredictorVectorType](
      this,
      "baseLearner",
      "base learner that will get stacked with boosting")

  /** @group getParam */
  def getBaseLearner: PredictorVectorType = $(baseLearner)

  /**
   * param for ratio of rows sampled out of the dataset
   *
   * @group param
   */
  val learningRate: Param[Double] =
    new DoubleParam(this, "learningRate", "learning rate for computing booster weights")

  /** @group getParam */
  def getLearningRate: Double = $(learningRate)

  setDefault(learningRate -> 0.5)

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
   * Loss function which Boosting tries to minimize. (case-insensitive)
   * Supported: "exponential"
   * (default = exponential)
   *
   * @group param
   */
  val loss: Param[String] =
    new Param(
      this,
      "loss",
      "loss function, exponential by default",
      (value: String) =>
        BoostingParams.supportedLossTypes.contains(value.toLowerCase(Locale.ROOT)))

  /** @group getParam */
  def getLoss: String = $(loss).toLowerCase(Locale.ROOT)

  setDefault(loss -> "exponential")

  setDefault(maxIter -> 10)

}

object BoostingParams {

  final val supportedLossTypes: Array[String] =
    Array("exponential", "squared", "absolute").map(_.toLowerCase(Locale.ROOT))

  def lossFunction(loss: String): Double => Double = loss match {
    case "exponential" => { error =>
      1 - FastMath.exp(-error)
    }
    case _ => throw new RuntimeException(s"Boosting was given bad loss type: $loss")

  }

  def saveImpl(
      path: String,
      instance: BoostingParams,
      sc: SparkContext,
      extraMetadata: Option[JObject] = None): Unit = {

    val params = instance.extractParamMap().toSeq
    val jsonParams = render(
      params
        .filter { case ParamPair(p, v) => p.name != "baseLearner" }
        .map { case ParamPair(p, v) => p.name -> parse(p.jsonEncode(v)) }
        .toList)

    DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, Some(jsonParams))

    val learnerPath = new Path(path, "learner").toString
    instance.getBaseLearner.asInstanceOf[MLWritable].save(learnerPath)

  }

  def loadImpl(
      path: String,
      sc: SparkContext,
      expectedClassName: String): (DefaultParamsReader.Metadata, PredictorVectorType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val learnerPath = new Path(path, "learner").toString
    val learner = DefaultParamsReader.loadParamsInstance[PredictorVectorType](learnerPath, sc)
    (metadata, learner)
  }

}
