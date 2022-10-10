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
// import org.apache.spark.sql.functions._
// import org.scalatest.FunSuite
// import org.apache.spark.ml.linalg.Vectors

// class GBMRegressorSuite extends FunSuite with DatasetSuiteBase {

//   test("benchmark") {

//     val raw = spark.read.format("libsvm").load("data/cpusmall/cpusmall.svm")

//     val tree =
//       new DecisionTreeRegressor()
//     val gmbr = new GBMRegressor()
//       .setBaseLearner(tree)
//     val gbt =
//       new GBTRegressor()

//     val re = new RegressionEvaluator()
//       .setLabelCol("label")
//       .setPredictionCol("prediction")
//       .setMetricName("rmse")

//     val data =
//       raw.withColumn("val", when(rand() > 0.8, true).otherwise(false))
//     data.cache().first()

//     time {
//       val gbmrParamGrid = new ParamGridBuilder()
//         .addGrid(gmbr.learningRate, Array(0.1))
//         .addGrid(gmbr.numBaseLearners, Array(30))
//         .addGrid(gmbr.validationIndicatorCol, Array("val"))
//         .addGrid(gmbr.tol, Array(1e-3))
//         .addGrid(gmbr.numRound, Array(8))
//         .addGrid(gmbr.sampleRatio, Array(0.8))
//         .addGrid(gmbr.replacement, Array(true))
//         .addGrid(gmbr.subspaceRatio, Array(0.8))
//         .addGrid(gmbr.optimizedWeights, Array(false, true))
//         .addGrid(gmbr.loss, Array("squared"))
//         .addGrid(gmbr.alpha, Array(0.5))
//         .build()

//       val gmbrCV = new CrossValidator()
//         .setEstimator(gmbr)
//         .setEvaluator(re)
//         .setEstimatorParamMaps(gbmrParamGrid)
//         .setNumFolds(3)
//         .setParallelism(4)

//       val gbmrCVModel = gmbrCV.fit(data)

//       println(gbmrCVModel.avgMetrics.mkString(","))
//       println(gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel].getLearningRate)
//       println(gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel].models.length)
//       println(gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel].const)
//       println(gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel].weights.mkString(","))
//       println(gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel].getLoss)
//       println(gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel].getOptimizedWeights)
//       println(gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel].getSampleRatio)
//       println(gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel].getReplacement)
//       println(gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel].getSubspaceRatio)
//       println(gbmrCVModel.avgMetrics.min)

//       val bm = gbmrCVModel.bestModel.asInstanceOf[GBMRegressionModel]
//       bm.write.overwrite().save("/tmp/bonjour")
//       val loaded = GBMRegressionModel.load("/tmp/bonjour")
//       assert(re.evaluate(loaded.transform(data)) == re.evaluate(bm.transform(data)))

//     }

//     time {
//       val paramGrid = new ParamGridBuilder()
//         .addGrid(gbt.stepSize, Array(0.1))
//         .addGrid(gbt.maxDepth, Array(10))
//         .addGrid(gbt.maxIter, Array(30))
//         .addGrid(gbt.subsamplingRate, Array(0.8))
//         .addGrid(gbt.validationIndicatorCol, Array("val"))
//         .addGrid(gbt.lossType, Array("squared", "absolute"))
//         .build()

//       val cv = new CrossValidator()
//         .setEstimator(gbt)
//         .setEvaluator(re)
//         .setEstimatorParamMaps(paramGrid)
//         .setNumFolds(3)
//         .setParallelism(4)

//       val cvModel = cv.fit(data)

//       println(cvModel.avgMetrics.mkString(","))
//       println(cvModel.bestModel.asInstanceOf[GBTRegressionModel].getLossType)
//       println(cvModel.bestModel.asInstanceOf[GBTRegressionModel].getNumTrees)
//       println(cvModel.bestModel.asInstanceOf[GBTRegressionModel].getSubsamplingRate)
//       println(cvModel.bestModel.asInstanceOf[GBTRegressionModel].getStepSize)
//       println(cvModel.avgMetrics.min)
//     }

//     time {
//       val paramGrid = new ParamGridBuilder()
//         .build()

//       val cv = new CrossValidator()
//         .setEstimator(tree)
//         .setEvaluator(
//           new RegressionEvaluator()
//             .setLabelCol(gbt.getLabelCol)
//             .setPredictionCol(gbt.getPredictionCol)
//             .setMetricName("rmse"))
//         .setEstimatorParamMaps(paramGrid)
//         .setNumFolds(3)
//         .setParallelism(4)

//       val cvModel = cv.fit(data)

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

//   test("trivial taks") {
//     val dr = new DecisionTreeRegressor()
//     val br = new GBMRegressor()
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
