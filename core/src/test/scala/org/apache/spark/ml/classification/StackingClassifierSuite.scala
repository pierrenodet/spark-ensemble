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

// package org.apache.spark.ml.classification

// import com.holdenkarau.spark.testing.DatasetSuiteBase
// import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
// import org.scalatest.FunSuite

// class StackingClassifierSuite extends FunSuite with DatasetSuiteBase {

//   test("benchmark") {

//     val raw = spark.read.format("libsvm").load("data/vehicle/vehicle.svm")

//     val sr = new StackingClassifier()
//       .setStacker(new DecisionTreeClassifier())
//       .setBaseLearners(Array(new DecisionTreeClassifier(), new RandomForestClassifier()))
//       .setParallelism(4)
//     val rf = new RandomForestClassifier()

//     val mce = new MulticlassClassificationEvaluator()
//       .setLabelCol("label")
//       .setPredictionCol("prediction")

//     val data = raw
//     data.cache()

//     time {
//       val srParamGrid = new ParamGridBuilder()
//         .build()

//       val srCV = new CrossValidator()
//         .setEstimator(sr)
//         .setEvaluator(mce)
//         .setEstimatorParamMaps(srParamGrid)
//         .setNumFolds(5)
//         .setParallelism(4)

//       val srCVModel = srCV.fit(data)

//       println(srCVModel.avgMetrics.max)

//       val bm = srCVModel.bestModel.asInstanceOf[StackingClassificationModel]
//       bm.write.overwrite().save("/tmp/bonjour")
//       val loaded = StackingClassificationModel.load("/tmp/bonjour")
//       assert(mce.evaluate(loaded.transform(data)) == mce.evaluate(bm.transform(data)))

//     }

//     time {
//       val paramGrid = new ParamGridBuilder()
//         .build()

//       val cv = new CrossValidator()
//         .setEstimator(rf)
//         .setEvaluator(mce)
//         .setEstimatorParamMaps(paramGrid)
//         .setNumFolds(5)
//         .setParallelism(4)

//       val cvModel = cv.fit(data)

//       println(cvModel.bestModel.asInstanceOf[RandomForestClassificationModel].getSubsamplingRate)
//       println(cvModel.avgMetrics.max)
//     }
//   }

//   def time[R](block: => R): R = {
//     val t0 = System.nanoTime()
//     val result = block // call-by-name
//     val t1 = System.nanoTime()
//     println("Elapsed time: " + (t1 - t0) + "ns")
//     result
//   }

// }
