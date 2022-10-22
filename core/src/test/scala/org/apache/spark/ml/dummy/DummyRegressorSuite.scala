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

package org.apache.spark.ml.dummy

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.scalacheck.Gen
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class DummyRegressorSuite
    extends AnyFunSuite
    with BeforeAndAfterAll
    with ScalaCheckPropertyChecks {

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
    spark.sparkContext.setCheckpointDir("checkpoint")
    spark.sparkContext.setLogLevel("ERROR")

  }

  override def afterAll() {
    spark.stop()
  }

  test("prediction is constant") {

    val gen = for {
      strategy <- Gen.oneOf(DummyRegressorParams.supportedStrategy)
      constant <- Gen.chooseNum(-100, 100)
      alpha <- Gen.chooseNum(0.05, 0.95)
    } yield (strategy, constant, alpha)

    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
    data.count()

    forAll(gen) { case (strategy, constant, alpha) =>
      val dummy =
        new DummyRegressor().setStrategy(strategy).setConstant(constant).setQuantile(alpha)
      val dummyModel = dummy.fit(data)
      val predictions =
        dummyModel.transform(data).select("prediction").collect().map(_.getDouble(0))

      assert(predictions.distinct.size == 1)
    }

  }

  test("const is equal to right statistics") {

    val gen = for {
      strategy <- Gen.oneOf(DummyRegressorParams.supportedStrategy)
      constant <- Gen.chooseNum(-100, 100)
      alpha <- Gen.chooseNum(0.05, 0.95)
    } yield (strategy, constant, alpha)

    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
    data.count()

    forAll(gen) { case (strategy, constant, alpha) =>
      val dummy =
        new DummyRegressor()
          .setStrategy(strategy)
          .setConstant(constant)
          .setQuantile(alpha)
          .setTol(0.0)

      val dummyModel = dummy.fit(data)
      val res = strategy match {
        case "mean" => data.select(mean("label")).first().getDouble(0)
        case "median" => data.stat.approxQuantile("label", Array(0.5), 0.0)(0)
        case "quantile" => data.stat.approxQuantile("label", Array(alpha), 0.0)(0)
        case "constant" => constant
      }

      assert(dummyModel.transform(data).select("prediction").first().getDouble(0) == res)
    }

  }

  test("read/write") {
    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dummy = new DummyRegressor()

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val dummyModel = dummy.fit(train)
    dummyModel.write.overwrite().save("/tmp/kek")
    val loaded = DummyRegressionModel.load("/tmp/kek")

    assert(dummyModel.transform(test).collect() === loaded.transform(test).collect())
  }

}
