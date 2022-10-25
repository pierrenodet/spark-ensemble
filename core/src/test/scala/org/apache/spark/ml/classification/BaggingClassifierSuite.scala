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
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import scala.collection.mutable.ListBuffer

class BaggingClassifierSuite extends AnyFunSuite with BeforeAndAfterAll {

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

  test("bagging classifier is better than baseline classifier") {

    val data =
      spark.read
        .format("libsvm")
        .load("data/letter/letter.svm")
        .withColumn("label", col("label") - lit(1))
        .cache()
    data.count()

    val dtc = new DecisionTreeClassifier()
    val bc = new BaggingClassifier()
      .setBaseLearner(dtc)
      .setNumBaseLearners(20)
      .setReplacement(true)
      .setSubsampleRatio(0.8)
      .setSubspaceRatio(0.8)
      .setParallelism(4)

    val mce = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val dtcModel = dtc.fit(train)
    val bcModel = bc.fit(train)

    assert(mce.evaluate(dtcModel.transform(test)) < mce.evaluate(bcModel.transform(test)))

  }

  test("bagging classifier is better than the best base classifier") {

    val data =
      spark.read
        .format("libsvm")
        .load("data/letter/letter.svm")
        .withColumn("label", col("label") - lit(1))
        .cache()
    data.count()

    val lr = new DecisionTreeClassifier()
    val bc = new BaggingClassifier()
      .setBaseLearner(lr)
      .setNumBaseLearners(20)
      .setReplacement(true)
      .setSubsampleRatio(0.8)
      .setSubspaceRatio(0.8)
      .setParallelism(4)

    val mce = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val bcModel = bc.fit(train)
    val metric = mce.evaluate(bcModel.transform(test))

    val metrics = ListBuffer.empty[Double]
    bcModel.models.foreach(model => metrics += mce.evaluate(model.transform(test)))

    assert(metrics.max < metric)
  }

  test("bagging classifier creates diversity among base classifiers") {

    val data =
      spark.read
        .format("libsvm")
        .load("data/letter/letter.svm")
        .withColumn("label", col("label") - lit(1))
        .cache()
    data.count()

    val lr = new DecisionTreeClassifier()
    val bc = new BaggingClassifier()
      .setBaseLearner(lr)
      .setNumBaseLearners(20)
      .setReplacement(true)
      .setSubsampleRatio(0.8)
      .setSubspaceRatio(0.8)
      .setParallelism(4)

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val bcModel = bc.fit(train)

    val mce = new MulticlassClassificationEvaluator()
      .setLabelCol("pred1")
      .setPredictionCol("pred2")
      .setMetricName("accuracy")

    val metrics = ListBuffer.empty[Double]
    val cross = bcModel.models.sliding(2)
    cross.foreach { case Array(model1, model2) =>
      val pred1UDF = udf { model1.predict(_) }
      val pred2UDF = udf { model2.predict(_) }
      val predDF = test
        .withColumn("pred1", pred1UDF(col("features")))
        .withColumn("pred2", pred2UDF(col("features")))
      metrics += mce.evaluate(predDF)
    }

    assert(metrics.max < 0.85)
  }

  test("read/write") {
    val data =
      spark.read
        .format("libsvm")
        .load("data/letter/letter.svm")
        .withColumn("label", col("label") - lit(1))
        .cache()
    data.count()

    val lr = new DecisionTreeClassifier()
    val bc = new BaggingClassifier()
      .setBaseLearner(lr)
      .setNumBaseLearners(4)
      .setReplacement(true)
      .setSubsampleRatio(0.4)
      .setParallelism(4)

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val bcModel = bc.fit(train)
    bcModel.write.overwrite().save("/tmp/baggingc")
    val loaded = BaggingClassificationModel.load("/tmp/baggingc")

    assert(bcModel.transform(test).collect() === loaded.transform(test).collect())
  }

}
