package org.apache.spark.ml.classification

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.scalatest.FunSuite

class BaggingClassifierSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read.option("header", "true").option("inferSchema", "true").csv("src/test/resources/data/iris/train.csv")

    val vectorAssembler = new VectorAssembler().setInputCols(raw.columns.filter(x => !x.equals("class"))).setOutputCol("features")
    val stringIndexer = new StringIndexer().setInputCol("class").setOutputCol("label")
    val br = new BaggingClassifier().setBaseLearner(new DecisionTreeClassifier()).setMaxIter(10).setParallelism(4)
    val rf = new RandomForestClassifier().setNumTrees(10)

    val data = stringIndexer.fit(raw).transform(vectorAssembler.transform(raw))
    data.count()

    time {
      val brParamGrid = new ParamGridBuilder()
        .addGrid(br.sampleRatioFeatures, Array(0.3,0.7,1))
        .addGrid(br.replacementFeatures, Array(x = false))
        .addGrid(br.replacement, Array(true, false))
        .addGrid(br.sampleRatio, Array(0.3, 0.7, 1))
        .build()

      val brCV = new CrossValidator()
        .setEstimator(br)
        .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol(br.getLabelCol).setPredictionCol(br.getPredictionCol))
        .setEstimatorParamMaps(brParamGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val brCVModel = brCV.fit(data)

      println(brCVModel.avgMetrics.mkString(","))
      print(brCVModel.bestModel.asInstanceOf[BaggingClassificationModel].getReplacement + ",")
      print(brCVModel.bestModel.asInstanceOf[BaggingClassificationModel].getSampleRatio + ",")
      print(brCVModel.bestModel.asInstanceOf[BaggingClassificationModel].getReplacementFeatures + ",")
      println(brCVModel.bestModel.asInstanceOf[BaggingClassificationModel].getSampleRatioFeatures)
      println(brCVModel.avgMetrics.max)

    }

    time {
      val paramGrid = new ParamGridBuilder()
        .addGrid(rf.subsamplingRate, Array(0.3, 0.7, 1))
        .build()

      val cv = new CrossValidator()
        .setEstimator(rf)
        .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol(rf.getLabelCol).setPredictionCol(rf.getPredictionCol))
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val cvModel = cv.fit(data)

      println(cvModel.avgMetrics.mkString(","))
      println(cvModel.bestModel.asInstanceOf[RandomForestClassificationModel].getSubsamplingRate)
      println(cvModel.avgMetrics.max)
    }
  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

}
