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

package org.apache.spark.ml.classification
import org.apache.spark._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

import scala.collection.mutable.ListBuffer

class GBMClassifierSuite extends AnyFunSuite with BeforeAndAfterAll {

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

  test("gbm classifier is better than baseline classifier") {

    val data =
      spark.read
        .format("libsvm")
        .load("data/letter/letter.svm")
        .withColumn("label", col("label") - lit(1))
        .cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
      .setMaxDepth(5)
    val gbmc = new GBMClassifier()
      .setBaseLearner(dtr)
      .setNumBaseLearners(3)
      .setLearningRate(1.0)
      .setUpdates("newton")
      .setParallelism(26)
    val dtc = new DecisionTreeClassifier()
      .setMaxDepth(5)
    val bc = new BoostingClassifier()
      .setBaseLearner(dtc)
      .setNumBaseLearners(3)

    val mce = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val dtcModel = spark.time(dtc.fit(train))
    val gbmcModel = spark.time(gbmc.fit(train))
    val bcModel = spark.time(bc.fit(train))

    assert(mce.evaluate(dtcModel.transform(test)) < mce.evaluate(gbmcModel.transform(test)))
    assert(mce.evaluate(bcModel.transform(test)) < mce.evaluate(gbmcModel.transform(test)))

  }

  test("gbm exponential and binomial binary classification is better than baselines") {

    val data =
      spark.read
        .format("libsvm")
        .load("data/adult/adult.svm")
        .withColumn("label", (col("label") + lit(1)) / lit(2.0))
        .cache()
    data.count()

    val maxDepth = 5
    val numBaseLearners = 10
    val dtr = new DecisionTreeRegressor()
      .setMaxDepth(maxDepth)
    val gbmcE = new GBMClassifier()
      .setBaseLearner(dtr)
      .setNumBaseLearners(numBaseLearners)
      .setLearningRate(1.0)
      .setLoss("exponential")
      .setUpdates("newton")
    val gbmcB = new GBMClassifier()
      .setBaseLearner(dtr)
      .setNumBaseLearners(numBaseLearners)
      .setLearningRate(1.0)
      .setLoss("binomial")
      .setUpdates("newton")
    val dtc = new DecisionTreeClassifier()
      .setMaxDepth(maxDepth)
    val bc = new BoostingClassifier()
      .setBaseLearner(dtc)
      .setNumBaseLearners(numBaseLearners)
    val gbtc = new GBTClassifier()
      .setMaxDepth(maxDepth)
      .setMaxIter(numBaseLearners)
      .setStepSize(1.0)

    val mce = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val dtcModel = spark.time(dtc.fit(train))
    val gbmcEModel = spark.time(gbmcE.fit(train))
    val gbmcBModel = spark.time(gbmcB.fit(train))
    val bcModel = spark.time(bc.fit(train))
    val gbtcModel = spark.time(gbtc.fit(train))

    assert(mce.evaluate(dtcModel.transform(test)) < mce.evaluate(gbmcBModel.transform(test)))
    assert(mce.evaluate(bcModel.transform(test)) < mce.evaluate(gbmcBModel.transform(test)))
    assert(mce.evaluate(gbtcModel.transform(test)) < mce.evaluate(gbmcBModel.transform(test)))
    assert(mce.evaluate(dtcModel.transform(test)) < mce.evaluate(gbmcEModel.transform(test)))
    assert(mce.evaluate(bcModel.transform(test)) < mce.evaluate(gbmcEModel.transform(test)))
    assert(mce.evaluate(gbtcModel.transform(test)) < mce.evaluate(gbmcEModel.transform(test)))
    assert(
      mce.evaluate(gbtcModel.transform(test)) === mce.evaluate(
        gbmcEModel.transform(test)) +- 0.05)

  }

  test("more base learners improves performance") {

    val data =
      spark.read
        .format("libsvm")
        .load("data/letter/letter.svm")
        .withColumn("label", col("label") - lit(1))
        .cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
      .setMaxDepth(5)
    val gbmc = new GBMClassifier()
      .setBaseLearner(dtr)
      .setNumBaseLearners(5)
      .setParallelism(26)
    val mce = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val gbmcModel = gbmc.fit(train)

    var metrics = ListBuffer.empty[Double]
    Array
      .range(1, gbmcModel.numModels + 1)
      .foreach(i => {
        val model = new GBMClassificationModel(
          gbmcModel.numClasses,
          gbmcModel.weights.take(i),
          gbmcModel.subspaces.take(i),
          gbmcModel.models.take(i),
          gbmcModel.init,
          gbmcModel.dim)
        metrics += mce.evaluate(model.transform(test))
      })

    assert(
      metrics.toList
        .sliding(2)
        .collect { case (h :: t) => h <= t.head }
        .count(identity) / (metrics.size - 1.0) >= 0.8)
  }

  test("early stop works") {

    val data =
      spark.read
        .format("libsvm")
        .load("data/adult/adult.svm")
        .withColumn("label", (col("label") + lit(1)) / lit(2.0))
        .withColumn("validation", when(rand() > 0.2, true).otherwise(false))
        .cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
    val gbmrWithVal = new GBMClassifier()
      .setBaseLearner(dtr)
      .setNumBaseLearners(20)
      .setLoss("binomial")
      .setUpdates("gradient")
      .setValidationIndicatorCol("validation")
      .setNumRounds(1)
      .setParallelism(26)

    val gbmrNoVal = new GBMClassifier()
      .setBaseLearner(dtr)
      .setNumBaseLearners(20)
      .setLoss("binomial")
      .setUpdates("gradient")
      .setParallelism(26)

    val mce = new MulticlassClassificationEvaluator().setMetricName("logLoss")

    val gbmrWithValModel = spark.time(gbmrWithVal.fit(data))
    val gbmrNoValModel = spark.time(gbmrNoVal.fit(data.filter(col("validation") === false)))

    val metrics = ListBuffer.empty[Double]
    Array
      .range(0, gbmrNoValModel.numModels + 1)
      .foreach(i => {
        val model =
          new GBMClassificationModel(
            gbmrNoValModel.numClasses,
            gbmrNoValModel.weights.take(i),
            gbmrNoValModel.subspaces.take(i),
            gbmrNoValModel.models.take(i),
            gbmrNoValModel.init,
            gbmrNoValModel.dim)
        metrics += mce.evaluate(model.transform(data.filter(col("validation") === true)))
      })

    val earlyStop = metrics.toList
      .sliding(2)
      .collect { case (h :: t) => h - t.head < 0.01 * math.max(0.01, t.head) }
      .indexOf(true)

    assert(gbmrWithValModel.numModels == earlyStop)

  }

  test("read/write") {
    val data =
      spark.read
        .format("libsvm")
        .load("data/letter/letter.svm")
        .withColumn("label", col("label") - lit(1))
        .cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
    val gbmc = new GBMClassifier()
      .setBaseLearner(dtr)
      .setNumBaseLearners(2)
      .setParallelism(26)

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val gbmcModel = gbmc.fit(train)
    gbmcModel.write.overwrite().save("/tmp/gbmc")
    val loaded = GBMClassificationModel.load("/tmp/gbmc")

    assert(gbmcModel.transform(test).collect() === loaded.transform(test).collect())
  }

  test("read/write exponential") {
    val data =
      spark.read
        .format("libsvm")
        .load("data/adult/adult.svm")
        .withColumn("label", (col("label") + lit(1)) / lit(2.0))
        .cache()
    data.count()

    val dtr = new DecisionTreeRegressor()
    val gbmc = new GBMClassifier()
      .setBaseLearner(dtr)
      .setLoss("exponential")
      .setNumBaseLearners(2)

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val gbmcModel = gbmc.fit(train)
    gbmcModel.write.overwrite().save("/tmp/gbmc")
    val loaded = GBMClassificationModel.load("/tmp/gbmc")

    assert(gbmcModel.transform(test).collect() === loaded.transform(test).collect())
  }

}
