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
import org.apache.spark.ml.impl.Utils.log1pExp
import org.apache.spark.ml.impl.Utils.softmax
import org.apache.spark.ml.linalg.BLAS
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.optim.aggregator.DifferentiableLossAggregator

private[spark] case class GBMLossInstance(
    label: Array[Double],
    weight: Double,
    prediction: Array[Double],
    direction: Array[Double])

class GBMLossAggregator(gbmLoss: GBMLoss)(bcCoefficients: Broadcast[Vector])
    extends DifferentiableLossAggregator[GBMLossInstance, GBMLossAggregator] {

  override protected val dim: Int = gbmLoss.dim

  @transient private lazy val coefficients = bcCoefficients.value match {
    case DenseVector(value) => value
    case _ =>
      throw new IllegalArgumentException(
        s"coefficients only supports dense vector but " +
          s"got type ${bcCoefficients.value.getClass}.)")
  }

  // Buffer for current predictions
  @transient private var buffer: Array[Double] = _

  def add(instance: GBMLossInstance): this.type = {
    if (buffer == null || buffer.length < dim) {
      buffer = Array.ofDim[Double](dim)
    }
    val arr = buffer
    var i = 0
    while (i < dim) {
      arr(i) = instance.prediction(i) + coefficients(i) * instance.direction(i)
      i += 1
    }
    i = 0
    while (i < dim) {
      lossSum += gbmLoss.loss(instance.label, arr)
      i += 1
    }
    weightSum += instance.weight
    i = 0
    val grad = gbmLoss.gradient(instance.label, arr)
    while (i < dim) {
      gradientSumArray(i) = gradientSumArray(i) + instance
        .direction(i) * grad(i)
      i += 1
    }
    this
  }

}

trait GBMLoss extends Serializable {

  def encodeLabel(label: Double): Array[Double]

  def dim: Int

  def loss(label: Array[Double], prediction: Array[Double]): Double

  def gradient(label: Array[Double], prediction: Array[Double]): Array[Double]

  def negativeGradient(label: Array[Double], prediction: Array[Double]): Array[Double] = {
    val grad = gradient(label, prediction)
    BLAS.getBLAS(dim).dscal(dim, -1.0, grad, 1)
    grad
  }

}

trait HasHessian {
  def hessian(label: Array[Double], prediction: Array[Double]): Array[Double]
}

trait HasScalarHessian extends HasHessian {
  def hessian(label: Double, prediction: Double): Double

  override def hessian(label: Array[Double], prediction: Array[Double]): Array[Double] = Array(
    hessian(label(0), prediction(0)))
}

trait GBMScalarLoss extends GBMLoss {
  override def dim: Int = 1

  def loss(label: Double, prediction: Double): Double

  def gradient(label: Double, prediction: Double): Double

  def negativeGradient(label: Double, prediction: Double): Double = -gradient(label, prediction)

  override def loss(label: Array[Double], prediction: Array[Double]): Double =
    loss(label(0), prediction(0))

  override def gradient(label: Array[Double], prediction: Array[Double]): Array[Double] = Array(
    gradient(label(0), prediction(0)))

}

trait GBMRegressionLoss extends GBMScalarLoss {
  override def encodeLabel(label: Double): Array[Double] = Array(label)

}

case object SquaredLoss extends GBMRegressionLoss with HasScalarHessian {
  def loss(label: Double, prediction: Double): Double =
    math.pow(label - prediction, 2) / 2.0

  def gradient(label: Double, prediction: Double): Double = -(label - prediction)

  def hessian(label: Double, prediction: Double): Double = 1.0

}

case object AbsoluteLoss extends GBMRegressionLoss {
  def loss(label: Double, prediction: Double): Double = math.abs(label - prediction)

  def gradient(label: Double, prediction: Double): Double = -math.signum(label - prediction)
}

case object LogCoshLoss extends GBMRegressionLoss with HasScalarHessian {
  def loss(label: Double, prediction: Double): Double = math.log(math.cosh(label - prediction))

  def gradient(label: Double, prediction: Double): Double = -math.tanh(label - prediction)

  def hessian(label: Double, prediction: Double): Double =
    1.0 / math.pow(math.cosh(label - prediction), 2)
}

case class ScaledLogCoshLoss(alpha: Double) extends GBMRegressionLoss with HasScalarHessian {
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

case class HuberLoss(delta: Double) extends GBMRegressionLoss {
  def loss(label: Double, prediction: Double): Double =
    if (math.abs(label - prediction) <= delta) math.pow(label - prediction, 2) / 2.0
    else delta * (math.abs(label - prediction) - delta / 2.0)

  def gradient(label: Double, prediction: Double): Double =
    if (math.abs(label - prediction) <= delta) -(label - prediction)
    else -delta * math.signum(label - prediction)

}

case class QuantileLoss(quantile: Double) extends GBMRegressionLoss {

  def loss(label: Double, prediction: Double): Double =
    if (label > prediction) quantile * (label - prediction)
    else (quantile - 1.0) * (label - prediction)

  def gradient(label: Double, prediction: Double): Double =
    if (label > prediction) -quantile else (1.0 - quantile)

}

trait GBMClassificationLoss extends GBMLoss {

  def raw2probabilityInPlace(rawPrediction: Vector): Vector

}

case class LogLoss(numClasses: Int) extends GBMClassificationLoss with HasHessian {

  override def dim: Int = numClasses

  override def encodeLabel(label: Double): Array[Double] = {
    val res = Array.fill(numClasses)(0.0)
    res(label.toInt) = 1.0
    res
  }

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
      res(i) = res(i) * (1 - res(i))
      i += 1
    }
    res
  }

  def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    softmax(rawPrediction.toArray)
    rawPrediction
  }

}

case object ExponentialLoss
    extends GBMClassificationLoss
    with GBMScalarLoss
    with HasScalarHessian {

  override def dim: Int = 1

  override def encodeLabel(label: Double): Array[Double] =
    Array(2 * label - 1)

  override def loss(label: Double, prediction: Double): Double =
    math.exp(-label * prediction)

  override def gradient(label: Double, prediction: Double): Double =
    -label * math.exp(-label * prediction)

  override def hessian(label: Double, prediction: Double): Double =
    math.pow(label, 2) * math.exp(-label * prediction)

  def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    val proba = Array.ofDim[Double](2)
    proba(1) = 1.0 / (1.0 + math.exp(-2.0 * rawPrediction(0)))
    proba(0) = 1.0 - proba(1)
    Vectors.dense(proba)
  }

}

case object BernoulliLoss extends GBMClassificationLoss with GBMScalarLoss with HasScalarHessian {

  override def dim: Int = 1

  override def encodeLabel(label: Double): Array[Double] =
    Array(2 * label - 1)

  override def loss(label: Double, prediction: Double): Double =
    log1pExp(-2 * label * prediction)

  override def gradient(label: Double, prediction: Double): Double =
    -2 * label / (1 + math.exp(2 * label * prediction))

  override def hessian(label: Double, prediction: Double): Double =
    (4 * math.exp(2 * prediction * label) * math.pow(label, 2)) / math.pow(
      1 + math.exp(2 * prediction * label),
      2)

  def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    val proba = Array.ofDim[Double](2)
    proba(1) = 1.0 / (1.0 + math.exp(rawPrediction(0)))
    proba(0) = 1.0 - proba(1)
    Vectors.dense(proba)
  }

}
