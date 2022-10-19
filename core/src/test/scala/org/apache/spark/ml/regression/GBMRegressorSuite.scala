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
import org.scalacheck.Gen
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import scala.collection.mutable.ListBuffer

class GBMRegressorSuite extends AnyFunSuite with BeforeAndAfterAll with ScalaCheckPropertyChecks {

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

  test("benchmark") {

    val raw =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm")
    val data = Seq.fill(20)(raw).reduce(_ union _).repartition(50).checkpoint()
    data.count()

    val n = 30
    val dtr = new DecisionTreeRegressor()
    val gbmr = new GBMRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(n)
      .setLearningRate(1.0)
      .setUpdates("newton")
      .setLoss("logcosh")
    val gbmrNoOptim = new GBMRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(n)
      .setLearningRate(1.0)
      // .setOptimizedWeights(false)
      .setUpdates("gradient")
      .setLoss("logcosh")
    val gbtr = new GBTRegressor().setMaxIter(n).setLossType("absolute")

    val re = new RegressionEvaluator().setMetricName("mae")

    val splits = data.randomSplit(Array(0.7, 0.3), 0L)
    val (train, test) = (splits(0), splits(1))

    val dtrModel = spark.time(dtr.fit(train))
    val gbmrModel = spark.time(gbmr.fit(train))
    val gbmrNoOptimModel = spark.time(gbmrNoOptim.fit(train))

    val gbmrMetrics = ListBuffer.empty[Double]
    val gbmrNoOptimMetrics = ListBuffer.empty[Double]
    val gbtrMetrics = ListBuffer.empty[Double]
    Array
      .range(0, gbmrModel.numBaseModels + 1)
      .foreach(i => {
        val model =
          new GBMRegressionModel(
            gbmrModel.weights.take(i),
            gbmrModel.subspaces.take(i),
            gbmrModel.models.take(i),
            gbmrModel.const)
        gbmrMetrics += re.evaluate(model.transform(test))
      })
    println(gbmrMetrics)
    println(gbmrModel.const, gbmrModel.weights.mkString(","))
    Array
      .range(0, gbmrNoOptimModel.numBaseModels + 1)
      .foreach(i => {
        val modelNoOptim =
          new GBMRegressionModel(
            gbmrNoOptimModel.weights.take(i),
            gbmrNoOptimModel.subspaces.take(i),
            gbmrNoOptimModel.models.take(i),
            gbmrNoOptimModel.const)
        gbmrNoOptimMetrics += re.evaluate(modelNoOptim.transform(test))
      })

    println(gbmrNoOptimMetrics)
    println(gbmrNoOptimModel.const, gbmrNoOptimModel.weights.mkString(","))

    val gbtrModel = spark.time(gbtr.fit(train))

    Array
      .range(1, gbtrModel.trees.size + 1)
      .foreach(i => {
        val model =
          new GBTRegressionModel(
            gbtrModel.uid,
            gbtrModel.trees.take(i),
            gbtrModel.treeWeights.take(i))
        gbtrMetrics += re.evaluate(model.transform(test))
      })
    println(gbtrMetrics)

    assert(re.evaluate(dtrModel.transform(test)) > re.evaluate(gbmrModel.transform(test)))
    assert(re.evaluate(gbtrModel.transform(test)) > re.evaluate(gbmrModel.transform(test)))
    assert(re.evaluate(gbmrNoOptimModel.transform(test)) > re.evaluate(gbmrModel.transform(test)))

  }

  test("boosting regressor is better than baseline regressor") {

    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()

    data.count()

    val dtr = new DecisionTreeRegressor()
    val gbmr = new GBMRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(10)
      .setLearningRate(0.4)
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
        .load("data/cpusmall/cpusmall.svm")
        .withColumn("validation", when(rand() > 0.2, true).otherwise(false))
        .cache()

    data.count()

    val dtr = new DecisionTreeRegressor()
    val gbmrWithVal = new GBMRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(100)
      .setValidationIndicatorCol("validation")
      .setNumRounds(1)
    val gbmrNoVal = new GBMRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(100)

    val re = new RegressionEvaluator().setMetricName("mse")

    val gbmrWithValModel = spark.time(gbmrWithVal.fit(data))
    val gbmrNoValModel = spark.time(gbmrNoVal.fit(data.filter(col("validation") === false)))

    val metrics = ListBuffer.empty[Double]
    Array
      .range(0, gbmrNoValModel.numBaseModels + 1)
      .foreach(i => {
        val model =
          new GBMRegressionModel(
            gbmrNoValModel.weights.take(i),
            gbmrNoValModel.subspaces.take(i),
            gbmrNoValModel.models.take(i),
            gbmrNoValModel.const)
        metrics += re.evaluate(model.transform(data.filter(col("validation") === true)))
      })

    val earlyStop = metrics.toList
      .sliding(2)
      .collect { case (h :: t) => h - t.head < 0.01 * math.max(0.01, t.head) }
      .indexOf(true)

    assert(gbmrWithValModel.numBaseModels == earlyStop)

  }

  test("more base learners improves performance") {

    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
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
      .range(0, gbmrModel.numBaseModels + 1)
      .foreach(i => {
        val model =
          new GBMRegressionModel(
            gbmrModel.weights.take(i),
            gbmrModel.subspaces.take(i),
            gbmrModel.models.take(i),
            gbmrModel.const)
        metrics += re.evaluate(model.transform(test))
      })

    assert(
      metrics.toList
        .sliding(2)
        .collect { case (h :: t) => h >= t.head }
        .count(identity) / (metrics.size - 1.0) == 1.0)

  }

  test("const is equal to right statistics") {

    val gen = for {
      loss <- Gen.oneOf(Seq("absolute"))
      alpha <- Gen.chooseNum(0.05, 0.95)
    } yield (loss, alpha)

    forAll(gen) { case (loss, alpha) =>
      val data =
        spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
      data.count()

      val dtr = new LinearRegression()
      val gbmr = new GBMRegressor()
        .setBaseLearner(dtr)
        .setNumBaseLearners(1)
        .setLoss(loss)
        .setAlpha(alpha)

      val gbmrModel = spark.time(gbmr.fit(data))
      val (ref, precision) = loss match {
        case "squared" => (data.select(mean("label")).first().getDouble(0), 1e-4)
        case "absolute" => (data.stat.approxQuantile("label", Array(0.5), 0.0)(0), 1e-4)
        case "quantile" => (data.stat.approxQuantile("label", Array(alpha), 0.0)(0), 1e-4)
        case "huber" => {
          val trim = if (alpha >= 0.5) alpha else 1 - alpha
          val q1 = data.stat.approxQuantile("label", Array(trim), 0.0)(0)
          val q2 = data.stat.approxQuantile("label", Array(1 - trim), 0.0)(0)
          (
            data
              .filter(col("label") <= q1 && col("label") >= q2)
              .select(mean("label"))
              .first()
              .getDouble(0),
            6.0)
        }

      }

      assert(gbmrModel.const === ref +- precision)
    }

  }

  test("read/write") {
    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
    val br = new GBMRegressor()
      .setBaseLearner(dtr)
      .setNumBaseLearners(5)

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val brModel = br.fit(train)
    brModel.write.overwrite().save("/tmp/kek")
    val loaded = GBMRegressionModel.load("/tmp/kek")

    assert(brModel.transform(test).collect() === loaded.transform(test).collect())
  }

}
