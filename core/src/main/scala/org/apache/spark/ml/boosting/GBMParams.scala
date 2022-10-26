/*
 * Copyright 2019 Pierre Nodet
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.boosting

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.ensemble._
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamValidators
import org.apache.spark.ml.param.shared._
import java.util.Locale

private[ml] trait GBMParams
    extends PredictorParams
    with HasMaxIter
    with HasTol
    with HasValidationIndicatorCol
    with BoostingParams[EnsembleRegressorType]
    with HasSubBag {

  /**
   * param for using optimized weights in GBM
   *
   * @group param
   */
  val optimizedWeights: Param[Boolean] =
    new BooleanParam(
      this,
      "optimizedWeights",
      "whether weights are optimized to minimize loss for each baseModel or weights are fixed to 1")

  /** @group getParam */
  def getOptimizedWeights: Boolean = $(optimizedWeights)

  /**
   * Newton (using hessian) or Gradient updates. (case-insensitive) Supported: "gradient",
   * "newton". (default = gradient)
   *
   * @group param
   */
  val updates: Param[String] =
    new Param(
      this,
      "updates",
      "updates, (case-insensitive). Supported options:" + s"${GBMParams.supportedUpdates
          .mkString(",")}",
      ParamValidators.inArray(GBMParams.supportedUpdates))

  /** @group getParam */
  def getUpdates: String = $(updates).toLowerCase(Locale.ROOT)

  /**
   * param for the learning rate of the algorithm
   *
   * @group param
   */
  val learningRate: Param[Double] =
    new DoubleParam(
      this,
      "learningRate",
      "learning rate for the estimator",
      ParamValidators.gt(0.0))

  /** @group getParam */
  def getLearningRate: Double = $(learningRate)

  /**
   * Threshold for stopping early when fit with validation is used. (This parameter is ignored
   * when fit without validation is used.) The decision to stop early is decided based on this
   * logic: If the current loss on the validation set is greater than 0.01, the diff of validation
   * error is compared to relative tolerance which is validationTol * (current loss on the
   * validation set). If the current loss on the validation set is less than or equal to 0.01, the
   * diff of validation error is compared to absolute tolerance which is validationTol * 0.01.
   * @group param
   * @see
   *   validationIndicatorCol
   */
  final val validationTol: DoubleParam = new DoubleParam(
    this,
    "validationTol",
    "Threshold for stopping early when fit with validation is used." +
      "If the error rate on the validation input changes by less than the validationTol," +
      "then learning will stop early (before `numBaseLearners`)." +
      "This parameter is ignored when fit without validation is used.",
    ParamValidators.gtEq(0.0))

  /** @group getParam */
  final def getValidationTol: Double = $(validationTol)

  /**
   * param for the number of round waiting for next decrease in validation set
   *
   * @group param
   */
  val numRounds: Param[Int] =
    new IntParam(
      this,
      "numRounds",
      "number of round waiting for next decrease in validation set",
      ParamValidators.gtEq(1))

  /** @group getParam */
  def getNumRounds: Int = $(numRounds)

  setDefault(optimizedWeights -> true)
  setDefault(updates -> "gradient")
  setDefault(learningRate -> 1.0)
  setDefault(numBaseLearners -> 10)
  setDefault(tol -> 1e-6)
  setDefault(maxIter -> 100)
  setDefault(numRounds -> 1)
  setDefault(validationTol -> 0.01)
  setDefault(replacement -> false)

}

private[ml] object GBMParams {
  final val supportedUpdates: Array[String] =
    Array("newton", "gradient").map(_.toLowerCase(Locale.ROOT))

}
