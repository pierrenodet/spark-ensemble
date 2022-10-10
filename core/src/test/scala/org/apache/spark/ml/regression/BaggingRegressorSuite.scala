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
import org.apache.spark.ml.Model
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import scala.collection.mutable.ListBuffer
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.BaggingRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.BaggingRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor

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

  }

  override def afterAll() {
    spark.stop()
  }

  test("bagging regressor is better than baseline regressor") {

    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dt = new DecisionTreeRegressor()
    val br = new BaggingRegressor()
      .setBaseLearner(dt)
      .setNumBaseLearners(5)
      .setReplacement(true)
      .setSubsampleRatio(0.6)
      .setParallelism(4)

    val re = new RegressionEvaluator().setMetricName("rmse")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val dtModel = dt.fit(train)
    val brModel = br.fit(train)

    assert(re.evaluate(dtModel.transform(test)) > re.evaluate(brModel.transform(test)))

  }

  test("bagging regressor is better than the best base regressor") {

    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dt = new DecisionTreeRegressor()
    val br = new BaggingRegressor()
      .setBaseLearner(dt)
      .setNumBaseLearners(10)
      .setReplacement(false)
      .setSubsampleRatio(0.6)
      .setParallelism(4)

    val re = new RegressionEvaluator().setMetricName("rmse")

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val brModel = br.fit(train)
    val metric = re.evaluate(brModel.transform(test))

    val metrics = ListBuffer.empty[Double]
    brModel.models.foreach(model => metrics += re.evaluate(model.transform(test)))

    assert(metrics.max > metric)
  }

  test("read/write") {
    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dt = new DecisionTreeRegressor()
    val br = new BaggingRegressor()
      .setBaseLearner(dt)
      .setNumBaseLearners(1)
      .setReplacement(true)
      .setSubsampleRatio(0.4)
      .setParallelism(4)

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val brModel = br.fit(train)
    brModel.write.overwrite().save("/tmp/kek")
    val loaded = BaggingRegressionModel.load("/tmp/kek")

    assert(brModel.transform(test).collect() === loaded.transform(test).collect())
  }

  test("bagging regressor with classifier should not work") {
    val data =
      spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm").cache()
    data.count()

    val dt = new DecisionTreeRegressor()
    val br = new BaggingRegressor()
      .setBaseLearner(dt)
      .setNumBaseLearners(1)
      .setReplacement(true)
      .setSubsampleRatio(0.4)
      .setParallelism(4)

    val splits = data.randomSplit(Array(0.8, 0.2), 0L)
    val (train, test) = (splits(0), splits(1))

    val brModel = br.fit(train)
    brModel.write.overwrite().save("/tmp/kek")
    val loaded = BaggingRegressionModel.load("/tmp/kek")

    assert(brModel.transform(test).collect() === loaded.transform(test).collect())
  }

}
