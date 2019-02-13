package org.apache.spark.ml.regression

import com.holdenkarau.spark.testing.DatasetSuiteBase
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.scalatest.FunSuite

class StackingRegressorSuite extends FunSuite with DatasetSuiteBase {

  test("benchmark") {

    val raw = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/test/resources/data/bostonhousing/train.csv")
    val test = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/test/resources/data/bostonhousing/test.csv")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(raw.columns.filter(x => !x.equals("ID") && !x.equals("medv")))
      .setOutputCol("features")
    val sr = new StackingRegressor()
      .setStacker(new DecisionTreeRegressor())
      .setLearners(Array(new DecisionTreeRegressor(), new RandomForestRegressor()))
      .setFeaturesCol("features")
      .setLabelCol("medv")
      .setParallelism(4)
    val rf = new RandomForestRegressor().setFeaturesCol("features").setLabelCol("medv").setNumTrees(10)

    val data = vectorAssembler.transform(raw)
    data.count()

    time {
      val srParamGrid = new ParamGridBuilder()
        .build()

      val srCV = new CrossValidator()
        .setEstimator(sr)
        .setEvaluator(new RegressionEvaluator().setLabelCol(sr.getLabelCol).setPredictionCol(sr.getPredictionCol))
        .setEstimatorParamMaps(srParamGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val srCVModel = srCV.fit(data)

      println(srCVModel.avgMetrics.min)

      val bm = srCVModel.bestModel.asInstanceOf[StackingRegressionModel]
      println(bm.explainParams())
      bm.write.overwrite().save("/tmp/bonjour")
      val loaded = StackingRegressionModel.load("/tmp/bonjour")
      println(loaded.explainParams())
      loaded.models.foreach(model => println(model.explainParams()))

    }

    time {
      val paramGrid = new ParamGridBuilder()
        .addGrid(rf.featureSubsetStrategy, Array("auto"))
        .addGrid(rf.numTrees, Array(10))
        .addGrid(rf.subsamplingRate, Array(0.3, 0.7, 1))
        .build()

      val cv = new CrossValidator()
        .setEstimator(rf)
        .setEvaluator(
          new RegressionEvaluator()
            .setLabelCol(rf.getLabelCol)
            .setPredictionCol(rf.getPredictionCol)
            .setMetricName("rmse")
        )
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5)
        .setParallelism(4)

      val cvModel = cv.fit(data)

      println(cvModel.avgMetrics.mkString(","))
      println(cvModel.bestModel.asInstanceOf[RandomForestRegressionModel].getSubsamplingRate)
      println(cvModel.avgMetrics.min)
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
