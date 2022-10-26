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
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import scala.collection.mutable.ListBuffer

class GBMRegressorSuite extends AnyFunSuite with BeforeAndAfterAll with ScalaCheckPropertyChecks {

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
      spark.read.format("libsvm").load("../data/cpusmall/cpusmall.svm").cache()

    data.count()

    val dtr = new DecisionTreeRegressor()
    val gbmr = new GBMRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(10)
    val gbtr = new GBTRegressor().setMaxIter(10)

    val re = new RegressionEvaluator().setMetricName("rmse")

    val splits = data.randomSplit(Array(0.7, 0.3), 0L)
    val (train, test) = (splits(0), splits(1))

    val dtrModel = spark.time(dtr.fit(train))
    val gbmrModel = spark.time(gbmr.fit(train))
    val gbtrModel = spark.time(gbtr.fit(train))

    assert(re.evaluate(dtrModel.transform(test)) > re.evaluate(gbmrModel.transform(test)))
    assert(re.evaluate(gbtrModel.transform(test)) > re.evaluate(gbmrModel.transform(test)))

  }

  test("early stop works") {

    val data =
      spark.read
        .format("libsvm")
        .load("../data/cpusmall/cpusmall.svm")
        .withColumn("validation", when(rand() > 0.2, true).otherwise(false))
        .cache()

    data.count()

    val dtr = new DecisionTreeRegressor()
    val gbmrWithVal = new GBMRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(10)
      .setValidationIndicatorCol("validation")
      .setNumRounds(1)
    val gbmrNoVal = new GBMRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(10)

    val re = new RegressionEvaluator().setMetricName("mse")

    val gbmrWithValModel = spark.time(gbmrWithVal.fit(data))
    val gbmrNoValModel = spark.time(gbmrNoVal.fit(data.filter(col("validation") === false)))

    val metrics = ListBuffer.empty[Double]
    Array
      .range(0, gbmrNoValModel.numModels + 1)
      .foreach(i => {
        val model =
          new GBMRegressionModel(
            gbmrNoValModel.weights.take(i),
            gbmrNoValModel.subspaces.take(i),
            gbmrNoValModel.models.take(i),
            gbmrNoValModel.init)
        metrics += re.evaluate(model.transform(data.filter(col("validation") === true)))
      })

    val earlyStop = metrics.toList
      .sliding(2)
      .collect { case (h :: t) => h - t.head < 0.01 * math.max(0.01, t.head) }
      .indexOf(true)

    assert(gbmrWithValModel.numModels == earlyStop)

  }

  test("more base learners improves performance") {

    val data =
      spark.read.format("libsvm").load("../data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
    val gbmr = new GBMRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(6)
      .setLearningRate(0.1)

    val re = new RegressionEvaluator().setMetricName("rmse")

    val splits = data.randomSplit(Array(0.7, 0.3), 0L)
    val (train, test) = (splits(0), splits(1))

    val gbmrModel = spark.time(gbmr.fit(train))

    val metrics = ListBuffer.empty[Double]
    Array
      .range(0, gbmrModel.numModels + 1)
      .foreach(i => {
        val model =
          new GBMRegressionModel(
            gbmrModel.weights.take(i),
            gbmrModel.subspaces.take(i),
            gbmrModel.models.take(i),
            gbmrModel.init)
        metrics += re.evaluate(model.transform(test))
      })

    assert(
      metrics.toList
        .sliding(2)
        .collect { case (h :: t) => h >= t.head }
        .count(identity) / (metrics.size - 1.0) == 1.0)

  }

  test("read/write") {
    val data =
      spark.read.format("libsvm").load("../data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
    val br = new GBMRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(5)

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val brModel = br.fit(train)
    brModel.write.overwrite().save("/tmp/gbmr")
    val loaded = GBMRegressionModel.load("/tmp/gbmr")

    assert(brModel.transform(test).collect() === loaded.transform(test).collect())
  }

}
