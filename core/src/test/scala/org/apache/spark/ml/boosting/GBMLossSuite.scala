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
import breeze.optimize.GradientTester
import org.apache.spark._
import org.apache.spark.ml.optim.loss.RDDLossFunction
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.sql._
import org.scalacheck.Gen
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

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

    val gen = for {
      loss <- Gen.oneOf(Seq(SquaredLoss, AbsoluteLoss, HuberLoss(0.9), QuantileLoss(0.9)))
    } yield (loss)

    forAll(gen) { case gbmLoss =>
      val labels = RandomRDDs.normalRDD(sc, 1000)
      val predictions = RandomRDDs.normalRDD(sc, 1000)
      val gbmInstances = labels.zip(predictions).map { case (label, prediciton) =>
        GBMInstance(label, 1.0, 0.0, prediciton)
      }
      val getAggregatorFunc = new GBMAggregator(gbmLoss)(_)

      val x = BDV[Double](1)
      val costFun =
        new RDDLossFunction(gbmInstances, getAggregatorFunc, None)
      assert(GradientTester.test[Int, BDV[Double]](costFun, x).apply(0) < 1e-6)
    }
  }

}
