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

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

import scala.collection.mutable.ListBuffer

class BoostingRegressorSuite extends AnyFunSuite with BeforeAndAfterAll {

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

  test("boosting regressor is better than baseline regressor") {

    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()

    data.count()

    val dtr = new DecisionTreeRegressor()
    val br = new BoostingRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(10)
    val gbtr = new GBTRegressor().setMaxIter(10)

    val re = new RegressionEvaluator().setMetricName("rmse")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val dtrModel = spark.time(dtr.fit(train))
    val brModel = spark.time(br.fit(train))
    val gbtrModel = spark.time(gbtr.fit(train))

    assert(re.evaluate(dtrModel.transform(test)) > re.evaluate(brModel.transform(test)))
    assert(re.evaluate(gbtrModel.transform(test)) > re.evaluate(brModel.transform(test)))

  }

  test("more base learners improves performance") {

    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dtr = new DecisionTreeRegressor().setMaxDepth(10)
    val br = new BoostingRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(10)

    val re = new RegressionEvaluator().setMetricName("rmse")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val brModel = br.fit(train)

    var metrics = ListBuffer.empty[Double]
    Array
      .range(1, brModel.numModels + 1)
      .foreach(i => {
        val model = new BoostingRegressionModel(brModel.weights.take(i), brModel.models.take(i))
        metrics += re.evaluate(model.transform(test))
      })

    assert(
      metrics.toList
        .sliding(2)
        .collect { case (h :: t) => h >= t.head }
        .count(identity) / (metrics.size - 1.0) >= 0.77)
  }

  test("weighted median is same as weighted mean") {

    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dtr = new DecisionTreeRegressor().setMaxDepth(10)
    val br = new BoostingRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(10)

    val re = new RegressionEvaluator().setMetricName("rmse")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val brModel = br.fit(train)
    val metric = re.evaluate(brModel.transform(test))
    val metricMean = re.evaluate(brModel.set(brModel.votingStrategy, "mean").transform(test))

    assert(metric === metricMean +- 0.1)
  }

  test("read/write") {
    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
    val br = new BoostingRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(5)

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val brModel = br.fit(train)
    brModel.write.overwrite().save("/tmp/kek")
    val loaded = BoostingRegressionModel.load("/tmp/kek")

    assert(brModel.transform(test).collect() === loaded.transform(test).collect())
  }

  test("maxErrorIsNull") {
    val dtr = new DecisionTreeRegressor()
    val br = new BoostingRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(20)
    val x = Seq.fill(100)(Vectors.dense(Array(1.0, 1.0)))
    val y = Seq.fill(100)(1.0)
    val data =
      spark.createDataFrame(spark.sparkContext.parallelize(x.zip(y))).toDF("features", "label")
    val learned = br.fit(data)
    val re = new RegressionEvaluator().setMetricName("rmse")
    assert(re.evaluate(learned.transform(data)) == 0.0)
    assert(learned.models.size < 20)
  }

  test("wrong label col throws error") {
    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
    val br = new BoostingRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(5)
      .setLabelCol("kek")

    assertThrows[IllegalArgumentException](br.fit(data))

  }

}
