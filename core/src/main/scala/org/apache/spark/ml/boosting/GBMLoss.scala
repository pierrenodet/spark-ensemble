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

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.BLAS
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.optim.aggregator.DifferentiableLossAggregator

/**
 * Class that represents an instance of weighted data point with label and features.
 *
 * @param label
 *   Label for this data point.
 * @param weight
 *   Weight for this data point.
 * @param prediction
 *   Prediction for this data point.
 */
private[spark] case class GBMInstance(
    label: Array[Double],
    weight: Double,
    prevPredciction: Array[Double],
    modelPrediction: Array[Double])

private[spark] object GBMInstance {

  def apply(label: Double, weight: Double, prevPredciction: Double, modelPrediction: Double) =
    new GBMInstance(Array(label), weight, Array(prevPredciction), Array(modelPrediction))

}

class GBMAggregator(loss: GBMLoss, override val dim: Int = 1)(bcCoefficients: Broadcast[Vector])
    extends DifferentiableLossAggregator[GBMInstance, GBMAggregator] {

  @transient private lazy val coefficients = bcCoefficients.value match {
    case DenseVector(value) => value
    case _ =>
      throw new IllegalArgumentException(
        s"coefficients only supports dense vector but " +
          s"got type ${bcCoefficients.value.getClass}.)")
  }

  // Buffer for current predictions
  @transient private var buffer: Array[Double] = _

  def add(instance: GBMInstance): this.type = {
    if (buffer == null || buffer.length < dim) {
      buffer = Array.ofDim[Double](dim)
    }
    val arr = buffer
    var i = 0
    while (i < dim) {
      arr(i) = instance.prevPredciction(i) + coefficients(0) * instance.modelPrediction(i)
      i += 1
    }
    lossSum += loss.loss(instance.label, arr)
    weightSum += instance.weight
    i = 0
    while (i < dim) {
      gradientSumArray(i) =
        gradientSumArray(i) + instance.modelPrediction(i) * loss.gradient(instance.label, arr)(i)
      i += 1
    }
    this
  }

}

trait GBMLoss extends Serializable {

  def loss(label: Array[Double], prediction: Array[Double]): Double

  def gradient(label: Array[Double], prediction: Array[Double]): Array[Double]

  def negativeGradient(label: Array[Double], prediction: Array[Double]): Array[Double] = {
    val grad = gradient(label, prediction)
    BLAS.getBLAS(grad.size).dscal(grad.size, -1.0, grad, 1)
    grad
  }

  def hessian(label: Array[Double], prediction: Array[Double]): Array[Double]

}

trait GBMScalarLoss extends GBMLoss {

  def loss(label: Double, prediction: Double): Double

  def loss(label: Array[Double], prediction: Array[Double]): Double =
    loss(label(0), prediction(0))

  def gradient(label: Double, prediction: Double): Double

  def gradient(label: Array[Double], prediction: Array[Double]): Array[Double] =
    Array(gradient(label(0), prediction(0)))

  def negativeGradient(label: Double, prediction: Double): Double = {
    -gradient(label, prediction)
  }

  def hessian(label: Double, prediction: Double): Double

  def hessian(label: Array[Double], prediction: Array[Double]): Array[Double] =
    Array(hessian(label(0), prediction(0)))
}

case object SquaredLoss extends GBMScalarLoss {
  def loss(label: Double, prediction: Double): Double =
    math.pow(label - prediction, 2) / 2.0

  def gradient(label: Double, prediction: Double): Double = -(label - prediction)

  def hessian(label: Double, prediction: Double): Double = 1.0

}

case object AbsoluteLoss extends GBMScalarLoss {
  def loss(label: Double, prediction: Double): Double = math.abs(label - prediction)

  def gradient(label: Double, prediction: Double): Double = -math.signum(label - prediction)

  def hessian(label: Double, prediction: Double): Double = 1.0
}

case object LogCoshLoss extends GBMScalarLoss {
  def loss(label: Double, prediction: Double): Double = math.log(math.cosh(label - prediction))

  def gradient(label: Double, prediction: Double): Double = -math.tanh(label - prediction)

  def hessian(label: Double, prediction: Double): Double =
    1.0 / math.pow(math.cosh(label - prediction), 2)
}

case class ScaledLogCoshLoss(alpha: Double) extends GBMScalarLoss {
  def loss(label: Double, prediction: Double): Double =
    if (label > prediction) alpha * LogCoshLoss.loss(label, prediction)
    else (1 - alpha) * LogCoshLoss.loss(label, prediction)

  def gradient(label: Double, prediction: Double): Double =
    if (label > prediction) alpha * LogCoshLoss.gradient(label, prediction)
    else (1 - alpha) * LogCoshLoss.gradient(label, prediction)

  def hessian(label: Double, prediction: Double): Double =
    if (label > prediction) alpha * LogCoshLoss.hessian(label, prediction)
    else (1 - alpha) * LogCoshLoss.hessian(label, prediction)
}

case class HuberLoss(delta: Double) extends GBMScalarLoss {
  def loss(label: Double, prediction: Double): Double =
    if (math.abs(label - prediction) <= delta) math.pow(label - prediction, 2) / 2.0
    else delta * (math.abs(label - prediction) - delta / 2.0)

  def gradient(label: Double, prediction: Double): Double =
    if (math.abs(label - prediction) <= delta) -(label - prediction)
    else -delta * math.signum(label - prediction)

  def hessian(label: Double, prediction: Double): Double =
    if (math.abs(label - prediction) <= delta) 1.0
    else delta

}

case class QuantileLoss(quantile: Double) extends GBMScalarLoss {

  def loss(label: Double, prediction: Double): Double =
    if (label > prediction) quantile * (label - prediction)
    else (quantile - 1.0) * (label - prediction)

  def gradient(label: Double, prediction: Double): Double =
    if (label > prediction) -quantile else (1.0 - quantile)

  def hessian(label: Double, prediction: Double): Double =
    if (label > prediction) -1.0
    else (1.0 - quantile)

}

case class LogLoss(numClasses: Int) extends GBMLoss {

  override def loss(label: Array[Double], prediction: Array[Double]): Double = {
    var i = 0
    var res = 0.0
    var sum = 0.0
    while (i < numClasses) {
      sum += math.exp(prediction(i))
      i += 1
    }
    val logsumexp = math.log(sum)
    i = 0
    while (i < numClasses) {
      res += -label(i) * (prediction(i) - logsumexp)
      i += 1
    }
    res
  }

  override def gradient(label: Array[Double], prediction: Array[Double]): Array[Double] = {
    var i = 0
    val res = Array.ofDim[Double](numClasses)
    var sum = 0.0
    while (i < numClasses) {
      sum += math.exp(prediction(i))
      i += 1
    }
    val logsumexp = math.log(sum)
    i = 0
    while (i < numClasses) {
      res(i) = math.exp(prediction(i) - logsumexp) - label(i)
      i += 1
    }
    res
  }

  override def hessian(label: Array[Double], prediction: Array[Double]): Array[Double] = {
    var i = 0
    val res = Array.ofDim[Double](numClasses)
    var sum = 0.0
    while (i < numClasses) {
      sum += math.exp(prediction(i))
      i += 1
    }
    val logsumexp = math.log(sum)
    i = 0
    while (i < numClasses) {
      res(i) = math.exp(prediction(i) - logsumexp)
      res(i) = math.max(2 * res(i) * (1 - res(i)), 1e-6)
      i += 1
    }
    res
  }

}
