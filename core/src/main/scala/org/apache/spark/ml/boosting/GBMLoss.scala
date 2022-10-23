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

import breeze.optimize.DiffFunction
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.impl.Utils.softmax
import org.apache.spark.ml.linalg.BLAS
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD

object GBMLineSearch {

  def functionFromSearchDirection(
      f: DiffFunction[RDD[Double]],
      x: RDD[Double],
      direction: RDD[Double]): DiffFunction[Double] =
    new DiffFunction[Double] {

      override def valueAt(alpha: Double): Double = {
        val newX = x.zip(direction).map { case (x, d) => x + alpha * d }
        f.valueAt(newX)
      }

      override def gradientAt(alpha: Double): Double = {
        val newX = x.zip(direction).map { case (x, d) => x + alpha * d }
        f.gradientAt(newX).zip(direction).map { case (g, d) => g * d }.mean
      }

      def calculate(alpha: Double): (Double, Double) = {
        val newX = x.zip(direction).map { case (x, d) => x + alpha * d }
        val (ff, grad) = f.calculate(newX)
        (ff, grad.zip(direction).map { case (g, d) => g * d }.mean)
      }
    }
}

class GBMDiffFunction(gbmLoss: GBMLoss, instances: RDD[Instance], aggregationDepth: Int = 2)
    extends DiffFunction[RDD[Double]]
    with Serializable {

  override def valueAt(x: RDD[Double]): Double = {
    val (sumLosses, sumWeights) = instances
      .zip(x)
      .treeAggregate((0.0, 0.0))(
        { case ((l, sw), (instance, x)) =>
          (l + gbmLoss.loss(instance.label, x), sw + instance.weight)
        },
        { case ((l1, s1), (l2, s2)) => (l1 + l2, s1 + s2) },
        aggregationDepth)
    sumLosses / sumWeights
  }

  override def gradientAt(x: RDD[Double]): RDD[Double] = {
    instances.zip(x).map { case (instance, x) => gbmLoss.gradient(instance.label, x) }
  }

  override def calculate(x: RDD[Double]): (Double, RDD[Double]) = {
    val lossGradWeight = instances.zip(x).map { case (instance, x) =>
      (gbmLoss.loss(instance.label, x), gbmLoss.gradient(instance.label, x), instance.weight)
    }
    val (sumLosses, sumWeights) = lossGradWeight.treeAggregate((0.0, 0.0))(
      { case ((sl, sw), (l, g, w)) =>
        (sl + l, sw + w)
      },
      { case ((l1, s1), (l2, s2)) => (l1 + l2, s1 + s2) },
      aggregationDepth)
    (sumLosses / sumWeights, lossGradWeight.map(_._2))
  }

}

trait GBMLoss extends Serializable {

  def loss(label: Double, prediction: Double): Double

  def gradient(label: Double, prediction: Double): Double

  def negativeGradient(label: Double, prediction: Double): Double = {
    -gradient(label, prediction)
  }

}

trait HasHessian {
  def hessian(label: Double, prediction: Double): Double
}

case object SquaredLoss extends GBMLoss with HasHessian {
  def loss(label: Double, prediction: Double): Double =
    math.pow(label - prediction, 2) / 2.0

  def gradient(label: Double, prediction: Double): Double = -(label - prediction)

  def hessian(label: Double, prediction: Double): Double = 1.0

}

case object AbsoluteLoss extends GBMLoss {
  def loss(label: Double, prediction: Double): Double = math.abs(label - prediction)

  def gradient(label: Double, prediction: Double): Double = -math.signum(label - prediction)
}

case object LogCoshLoss extends GBMLoss with HasHessian {
  def loss(label: Double, prediction: Double): Double = math.log(math.cosh(label - prediction))

  def gradient(label: Double, prediction: Double): Double = -math.tanh(label - prediction)

  def hessian(label: Double, prediction: Double): Double =
    1.0 / math.pow(math.cosh(label - prediction), 2)
}

case class ScaledLogCoshLoss(alpha: Double) extends GBMLoss with HasHessian {
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

case class HuberLoss(delta: Double) extends GBMLoss {
  def loss(label: Double, prediction: Double): Double =
    if (math.abs(label - prediction) <= delta) math.pow(label - prediction, 2) / 2.0
    else delta * (math.abs(label - prediction) - delta / 2.0)

  def gradient(label: Double, prediction: Double): Double =
    if (math.abs(label - prediction) <= delta) -(label - prediction)
    else -delta * math.signum(label - prediction)

}

case class QuantileLoss(quantile: Double) extends GBMLoss {

  def loss(label: Double, prediction: Double): Double =
    if (label > prediction) quantile * (label - prediction)
    else (quantile - 1.0) * (label - prediction)

  def gradient(label: Double, prediction: Double): Double =
    if (label > prediction) -quantile else (1.0 - quantile)

}

trait GBMClassificationLoss extends GBMLoss {

  def dim: Int

  def raw2probabilityInPlace(rawPrediction: Vector): Vector

  def encodeLabel(label: Double): Vector

}

case class LogLoss(numClasses: Int) extends GBMClassificationLoss with HasHessian {

  override def dim: Int = numClasses

  override def encodeLabel(label: Double): Vector =
    Vectors.sparse(numClasses, Seq((label.toInt, 1.0)))

  override def loss(label: Double, prediction: Double): Double =
    -label * math.log(prediction)

  override def gradient(label: Double, prediction: Double): Double =
    prediction - label

  override def hessian(label: Double, prediction: Double): Double =
    math.max(2 * prediction * (1 - prediction), 1e-6)

  def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    softmax(rawPrediction.toArray)
    rawPrediction
  }

}

case object ExponentialLoss extends GBMClassificationLoss with HasHessian {

  override def dim: Int = 1

  override def encodeLabel(label: Double): Vector =
    Vectors.dense(Array(2.0 * label - 1.0))

  override def loss(label: Double, prediction: Double): Double =
    math.exp(-label * prediction)

  override def gradient(label: Double, prediction: Double): Double =
    -label * math.exp(-label * prediction)

  override def hessian(label: Double, prediction: Double): Double =
    label * label * math.exp(-label * prediction)

  def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    val proba = Array.ofDim[Double](2)
    proba(1) = math.exp(2.0 * rawPrediction(0))
    proba(0) = 1.0 - proba(1)
    Vectors.dense(proba)
  }

}
