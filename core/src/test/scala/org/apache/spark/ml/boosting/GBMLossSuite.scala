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

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.DiffFunction
import breeze.optimize.GradientTester
import breeze.util.SerializableLogging
import org.apache.spark._
import org.apache.spark.ml.optim.loss.RDDLossFunction
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.sql._
import org.scalacheck.Gen
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

object GradientDoubleTesting extends SerializableLogging {
  def test(
      f: DiffFunction[Double],
      x: Double,
      skipZeros: Boolean = false,
      epsilon: Double = 1e-8,
      tolerance: Double = 1e-3): Double = {
    val (fx, trueGrad) = f.calculate(x)
    var differences = 0.0
    var xx = x
    if (skipZeros && trueGrad == 0.0) {} else {
      xx += epsilon
      val grad = (f(xx) - fx) / epsilon
      xx -= epsilon
      val relDif = (grad - trueGrad).abs / math.max(trueGrad.abs, grad.abs).max(1e-4)
      if (relDif < tolerance) {
        logger.debug(s"OK: $relDif")
      } else {
        logger.warn(
          "relDif: %.3e [eps : %e, calculated: %4.3e empirical: %4.3e]"
            .format(relDif, epsilon, trueGrad, grad))
      }
      differences = relDif
    }

    differences
  }
}

class GBMLossSuite extends AnyFunSuite with BeforeAndAfterAll with ScalaCheckPropertyChecks {

  import org.scalacheck.Shrink.shrinkAny

  var spark: SparkSession = _

  override def beforeAll() {

    spark = SparkSession
      .builder()
      .config(
        new SparkConf()
          .setMaster("local[*]")
          .setAppName("example"))
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

  }

  override def afterAll() {
    spark.stop()
  }

  test("verify losses and gradients") {

    val sc = spark.sparkContext

    val losses = Seq(
      SquaredLoss,
      AbsoluteLoss,
      HuberLoss(0.9),
      QuantileLoss(0.9),
      LogCoshLoss,
      ScaledLogCoshLoss(0.9))

    val gbmLossesWithHessian =
      losses.collect { case gbmLoss: HasScalarHessian =>
        new GBMRegressionLoss {
          override def loss(label: Double, prediction: Double): Double =
            gbmLoss.gradient(label, prediction)

          override def gradient(label: Double, prediction: Double): Double =
            gbmLoss.hessian(label, prediction)
        }
      }

    val gen = for {
      loss <- Gen.oneOf(losses ++ gbmLossesWithHessian)
    } yield (loss)

    forAll(gen) { case gbmLoss =>
      val labels = RandomRDDs.normalRDD(sc, 1000)
      val predictions = RandomRDDs.normalRDD(sc, 1000)
      val gbmInstances = labels.zip(predictions).map { case (label, prediciton) =>
        GBMLossInstance(gbmLoss.encodeLabel(label), 1.0, Array(0.0), Array(prediciton))
      }
      val getAggregatorFunc = new GBMLossAggregator(gbmLoss)(_)

      val x = BDV[Double](1)
      val costFun =
        new RDDLossFunction(gbmInstances, getAggregatorFunc, None)
      assert(GradientTester.test[Int, BDV[Double]](costFun, x).apply(0) < 1e-6)
    }

  }
}
