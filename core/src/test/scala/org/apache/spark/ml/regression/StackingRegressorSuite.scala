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

package org.apache.spark.ml.regression

import org.apache.spark._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import scala.collection.mutable.ListBuffer

class StackingRegressorSuite extends AnyFunSuite with BeforeAndAfterAll {

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

  test("stacking regressor is better than baseline regressors") {

    val data =
      spark.read.format("libsvm").load("../data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
    val boostingr = new BoostingRegressor().setNumBaseLearners(5).setBaseLearner(dtr)
    val gbmr =
      new GBMRegressor()
        .setNumBaseLearners(5)
        .setBaseLearner(dtr)
    val lr = new LinearRegression()
    val sc = new StackingRegressor()
      .setBaseLearners(Array(dtr, boostingr, gbmr, lr))
      .setStacker(lr)
      .setParallelism(4)

    val re = new RegressionEvaluator().setMetricName("rmse")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val lrModel = lr.fit(train)
    val scModel = sc.fit(train)

    assert(re.evaluate(lrModel.transform(test)) > re.evaluate(scModel.transform(test)))

  }

  test("stacking regressor is better than the best base regressor") {

    val data =
      spark.read.format("libsvm").load("../data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
    val boostingr = new BoostingRegressor().setNumBaseLearners(5).setBaseLearner(dtr)
    val gbmr =
      new GBMRegressor()
        .setNumBaseLearners(5)
        .setBaseLearner(dtr)
    val lr = new LinearRegression()
    val sc = new StackingRegressor()
      .setBaseLearners(Array(dtr, boostingr, gbmr, lr))
      .setStacker(lr)
      .setParallelism(4)

    val re = new RegressionEvaluator().setMetricName("rmse")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val scModel = sc.fit(train)

    val metric = re.evaluate(scModel.transform(test))

    val metrics = ListBuffer.empty[Double]
    scModel.models.foreach(model => metrics += re.evaluate(model.transform(test)))

    assert(metrics.min > metric)
  }

  test("read/write") {
    val data =
      spark.read.format("libsvm").load("../data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
    val lr = new LinearRegression()
    val sc = new StackingRegressor()
      .setBaseLearners(Array.fill(4)(dtr))
      .setStacker(lr)

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val scModel = sc.fit(train)
    scModel.write.overwrite().save("/tmp/sc")
    val loaded = StackingRegressionModel.load("/tmp/sc")

    assert(scModel.transform(test).collect() === loaded.transform(test).collect())
  }

}
