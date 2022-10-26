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
import org.apache.spark.ml.regression.DecisionTreeRegressor

class StackingClassifierSuite extends AnyFunSuite with BeforeAndAfterAll {

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

  test("stacking classifier is better than the best base classifier") {

    val data =
      spark.read
        .format("libsvm")
        .load("../data/letter/letter.svm")
        .withColumn("label", col("label") - lit(1))
        .cache()
    data.count()

    val dtc = new DecisionTreeClassifier()
    val boostingc = new BoostingClassifier().setNumBaseLearners(5).setBaseLearner(dtc)
    val gbmc =
      new GBMClassifier()
        .setNumBaseLearners(5)
        .setBaseLearner(new DecisionTreeRegressor())
        .setParallelism(26)
    val lr = new LogisticRegression()
    val sc = new StackingClassifier()
      .setBaseLearners(Array(dtc, boostingc, gbmc, lr))
      .setStacker(lr)
      .setStackMethod("raw")
      .setParallelism(4)

    val mce = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val scModel = sc.fit(train)

    val metric = mce.evaluate(scModel.transform(test))

    val metrics = ListBuffer.empty[Double]
    scModel.models.foreach(model => metrics += mce.evaluate(model.transform(test)))

    assert(metrics.max < metric)
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
    val lr = new LogisticRegression().setRegParam(0.5).setElasticNetParam(1.0)
    val sc = new StackingClassifier()
      .setBaseLearners(Array.fill(4)(dtc))
      .setStacker(lr)
      .setParallelism(4)

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val scModel = sc.fit(train)
    scModel.write.overwrite().save("/tmp/sc")
    val loaded = StackingClassificationModel.load("/tmp/sc")

    assert(scModel.transform(test).collect() === loaded.transform(test).collect())
  }

}
