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

// package org.apache.spark.ml.regression

// import com.holdenkarau.spark.testing.DatasetSuiteBase
// import org.apache.spark.ml.evaluation.RegressionEvaluator
// import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
// import org.apache.spark.sql.functions.{rand, when}
// import org.scalatest.FunSuite
// import org.apache.spark.ml.linalg.Vectors

// class BoostingRegressorSuite extends FunSuite with DatasetSuiteBase {

//   test("benchmark") {

//     val raw = spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm")

//     val dr = new DecisionTreeRegressor()
//     val br = new BoostingRegressor()
//       .setBaseLearner(dr)
//     val gbt = new GBTRegressor()

//     val re = new RegressionEvaluator()
//       .setLabelCol("label")
//       .setPredictionCol("prediction")
//       .setMetricName("rmse")

//     val data = raw.withColumn("val", when(rand() > 0.8, true).otherwise(false))
//     data.cache().first()

//     time {
//       val brParamGrid = new ParamGridBuilder()
//         .addGrid(br.loss, Array("squared", "exponential"))
//         .addGrid(br.validationIndicatorCol, Array("val"))
//         .addGrid(br.numBaseLearners, Array(20))
//         .addGrid(br.tol, Array(1e-9))
//         .addGrid(br.numRound, Array(3))
//         .addGrid(dr.maxDepth, Array(10))
//         .build()

//       val brCV = new CrossValidator()
//         .setEstimator(br)
//         .setEvaluator(re)
//         .setEstimatorParamMaps(brParamGrid)
//         .setNumFolds(3)
//         .setParallelism(4)

//       val brCVModel = brCV.fit(data)

//       println(brCVModel.avgMetrics.mkString(","))
//       println(brCVModel.bestModel.asInstanceOf[BoostingRegressionModel].getLoss)
//       println(brCVModel.bestModel.asInstanceOf[BoostingRegressionModel].models.length)
//       println(brCVModel.bestModel.asInstanceOf[BoostingRegressionModel].weights.mkString(","))
//       println(brCVModel.avgMetrics.min)

//       val bm = brCVModel.bestModel.asInstanceOf[BoostingRegressionModel]
//       bm.write.overwrite().save("/tmp/bonjour")
//       val loaded = BoostingRegressionModel.load("/tmp/bonjour")
//       assert(re.evaluate(loaded.transform(data)) == re.evaluate(bm.transform(data)))

//     }

//     time {
//       val paramGrid = new ParamGridBuilder()
//         .addGrid(gbt.featureSubsetStrategy, Array("auto"))
//         .addGrid(gbt.validationIndicatorCol, Array("val"))
//         .addGrid(gbt.subsamplingRate, Array(0.3, 0.5, 0.7, 1))
//         .addGrid(gbt.maxIter, Array(20))
//         .build()

//       val cv = new CrossValidator()
//         .setEstimator(gbt)
//         .setEvaluator(re)
//         .setEstimatorParamMaps(paramGrid)
//         .setNumFolds(3)
//         .setParallelism(4)

//       val cvModel = cv.fit(data)

//       println(cvModel.avgMetrics.mkString(","))
//       println(cvModel.bestModel.asInstanceOf[GBTRegressionModel].getSubsamplingRate)
//       println(cvModel.bestModel.asInstanceOf[GBTRegressionModel].getNumTrees)
//       println(cvModel.avgMetrics.min)
//     }
//   }

//   def time[R](block: => R): R = {
//     val t0 = System.nanoTime()
//     val result = block // call-by-name
//     val t1 = System.nanoTime()
//     println("Elapsed time: " + (t1 - t0) + "ns")
//     result
//   }

//   test("maxErrorIsNull") {
//     val dr = new DecisionTreeRegressor()
//     val br = new BoostingRegressor()
//       .setBaseLearner(dr)
//       .setNumBaseLearners(20)
//     val x = Seq.fill(100)(Vectors.dense(Array(1.0, 1.0)))
//     val y = Seq.fill(100)(1.0)
//     import spark.implicits._
//     val data = spark.sparkContext.parallelize(x.zip(y)).toDF("features", "label")
//     val learned = br.fit(data)
//     learned.transform(data).show()
//   }

// }
