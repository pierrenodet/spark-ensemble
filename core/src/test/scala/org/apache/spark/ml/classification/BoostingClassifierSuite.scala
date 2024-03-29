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
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

import scala.collection.mutable.ListBuffer
import org.apache.spark.ml.linalg.DenseVector

class BoostingClassifierSuite extends AnyFunSuite with BeforeAndAfterAll {

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

  test("more base learners improves performance") {

    val data =
      spark.read
        .format("libsvm")
        .load("../data/letter/letter.svm")
        .withColumn("label", col("label") - lit(1))
        .cache()
    data.count()

    val dtc = new DecisionTreeClassifier()
      .setMaxDepth(10)
    val bc = new BoostingClassifier()
      .setBaseLearner(dtc)
      .setNumBaseLearners(10)

    val mce = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    val splits = data.randomSplit(Array(0.7, 0.3), 0L)
    val (train, test) = (splits(0), splits(1))

    val bcModel = bc.fit(train)

    var metrics = ListBuffer.empty[Double]
    Array
      .range(1, bcModel.numModels + 1)
      .foreach(i => {
        val model = new BoostingClassificationModel(
          bcModel.numClasses,
          bcModel.weights.take(i),
          bcModel.models.take(i))
        metrics += mce.evaluate(model.transform(test))
      })

    assert(
      metrics.toList
        .sliding(2)
        .collect { case (h :: t) => h < t.head }
        .count(identity) / (metrics.size - 1.0) >= 0.8)
  }

  test("SAMME.R equivalent to SAMME") {

    val data =
      spark.read
        .format("libsvm")
        .load("../data/letter/letter.svm")
        .withColumn("label", col("label") - lit(1))
        .cache()
    data.count()

    val lr = new DecisionTreeClassifier().setMaxDepth(10)
    val bcr = new BoostingClassifier()
      .setBaseLearner(lr)
      .setNumBaseLearners(10)
      .setAlgorithm("real")
    val bcd = new BoostingClassifier()
      .setBaseLearner(lr)
      .setNumBaseLearners(10)
      .setAlgorithm("discrete")

    val mce = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    val splits = data.randomSplit(Array(0.7, 0.3), 0L)
    val (train, test) = (splits(0), splits(1))
    train.count()

    val bcrModel = spark.time(bcr.fit(train))
    val bcdModel = spark.time(bcd.fit(train))

    assert(
      mce.evaluate(bcrModel.transform(test)) === mce.evaluate(bcdModel.transform(test)) +- 0.02)
  }

  List("discrete", "real").foreach { algorithm =>
    test(
      f"$algorithm boosting decision function respects the symmetric constraint for weak learners") {
      val data =
        spark.read
          .format("libsvm")
          .load("../data/letter/letter.svm")
          .withColumn("label", col("label") - lit(1))
          .cache()
      data.count()
      val numClasses = data.select("label").distinct().count()

      val lr = new DecisionTreeClassifier().setMaxDepth(1)
      val boost = new BoostingClassifier()
        .setBaseLearner(lr)
        .setNumBaseLearners(10)
        .setAlgorithm(algorithm)
      val boostModel = boost.fit(data)

      val predictions = boostModel
        .transform(data)
        .select("rawPrediction")
        .distinct()
        .collect()
        .map(_.getAs[DenseVector](0).toArray)

      predictions.foreach(raw => assert(raw.sum === 0.0 +- 1e-6))
    }
  }

  test("read/write") {
    val data =
      spark.read
        .format("libsvm")
        .load("../data/letter/letter.svm")
        .withColumn("label", col("label") - lit(1))
        .cache()
    data.count()

    val dtc = new DecisionTreeClassifier()
    val bc = new BoostingClassifier()
      .setBaseLearner(dtc)
      .setNumBaseLearners(5)

    val splits = data.randomSplit(Array(0.7, 0.3), 0L)
    val (train, test) = (splits(0), splits(1))

    val bcModel = bc.fit(train)
    bcModel.write.overwrite().save("/tmp/boostingc")
    val loaded = BoostingClassificationModel.load("/tmp/boostingc")

    assert(bcModel.transform(test).collect() === loaded.transform(test).collect())
  }

}
