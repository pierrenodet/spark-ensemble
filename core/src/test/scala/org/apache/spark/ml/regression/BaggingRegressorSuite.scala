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

class BaggingRegressorSuite extends AnyFunSuite with BeforeAndAfterAll {

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
    spark.sparkContext.setCheckpointDir("checkpoint")

  }

  override def afterAll() {
    spark.stop()
  }

  test("bagging regressor is better than baseline regressor") {

    val data =
      spark.read.format("libsvm").load("../data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dt = new DecisionTreeRegressor()
    val br = new BaggingRegressor()
      .setBaseLearner(dt)
      .setNumBaseLearners(20)
      .setSubsampleRatio(0.7)
      .setSubspaceRatio(0.75)
      .setParallelism(20)
    val rf = new RandomForestRegressor().setNumTrees(20).setSubsamplingRate(0.7)

    val re = new RegressionEvaluator().setMetricName("rmse")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val dtModel = spark.time(dt.fit(train))
    val brModel = spark.time(br.fit(train))
    val rfModel = spark.time(rf.fit(train))

    assert(re.evaluate(dtModel.transform(test)) > re.evaluate(brModel.transform(test)))
    assert(re.evaluate(rfModel.transform(test)) > re.evaluate(brModel.transform(test)))

  }

  test("bagging regressor is better than the best base regressor") {

    val data =
      spark.read.format("libsvm").load("../data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dt = new DecisionTreeRegressor()
    val br = new BaggingRegressor()
      .setBaseLearner(dt)
      .setNumBaseLearners(10)
      .setReplacement(false)
      .setSubsampleRatio(0.6)
      .setParallelism(10)

    val re = new RegressionEvaluator().setMetricName("rmse")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val brModel = br.fit(train)
    val metric = re.evaluate(brModel.transform(test))

    val metrics = ListBuffer.empty[Double]
    brModel.models.foreach(model => metrics += re.evaluate(model.transform(test)))

    assert(metrics.min > metric)
  }

  test("read/write") {
    val data =
      spark.read.format("libsvm").load("../data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dt = new DecisionTreeRegressor()
    val br = new BaggingRegressor()
      .setBaseLearner(dt)
      .setNumBaseLearners(2)
      .setReplacement(true)
      .setSubsampleRatio(0.4)
      .setParallelism(2)

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val brModel = br.fit(train)
    brModel.write.overwrite().save("/tmp/baggingr")
    val loaded = BaggingRegressionModel.load("/tmp/baggingr")

    assert(brModel.transform(test).collect() === loaded.transform(test).collect())
  }

}
