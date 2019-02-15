package org.apache.spark.ml.ensemble
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasSeed

trait SubSpaceParams extends Params with HasSeed {

  /**
   * param for whether samples are drawn with replacement
   *
   * @group param
   */
  val replacement: Param[Boolean] =
    new BooleanParam(this, "replacement", "whether samples are drawn with replacement")

  /** @group getParam */
  def getReplacement: Boolean = $(replacement)

  setDefault(replacement -> false)

  /**
   * param for ratio of rows sampled out of the dataset
   *
   * @group param
   */
  val sampleRatio: Param[Double] =
    new DoubleParam(
      this,
      "sampleRatio",
      "ratio of rows sampled out of the dataset",
      ParamValidators.inRange(0, 1))

  /** @group getParam */
  def getSampleRatio: Double = $(sampleRatio)

  setDefault(sampleRatio -> 1)

  /**
   * param for whether samples are drawn with replacement
   *
   * @group param
   */
  val replacementFeatures: Param[Boolean] =
    new BooleanParam(
      this,
      "replacementFeautres",
      "whether features sampling are drawn with replacement")

  /** @group getParam */
  def getReplacementFeatures: Boolean = $(replacementFeatures)

  setDefault(replacementFeatures -> false)

  /**
   * param for ratio of rows sampled out of the dataset
   *
   * @group param
   */
  val sampleRatioFeatures: Param[Double] =
    new DoubleParam(
      this,
      "sampleRatioFeatures",
      "ratio of features sampled out of the dataset",
      ParamValidators.inRange(0, 1))

  /** @group getParam */
  def getSampleRatioFeatures: Double = $(sampleRatioFeatures)

  setDefault(sampleRatioFeatures -> 1)

}
