package org.apache.spark.ml.regression

import org.apache.commons.math3.distribution.PoissonDistribution
import org.apache.spark.SparkException
import org.apache.spark.ml.bagging.BaggingParams
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util.{BaggingMetadataUtils, Identifiable}
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.sql.bfunctions._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}
import org.apache.spark.util.ThreadUtils
import org.apache.spark.util.random.XORShiftRandom

import scala.concurrent.Future
import scala.concurrent.duration.Duration
import scala.util.Random

class BaggingRegressor(override val uid: String) extends Regressor[Vector, BaggingRegressor, BaggingRegressionModel] with BaggingParams {

  def this() = this(Identifiable.randomUID("BaggingRegressor"))

  // Parameters from BaggingRegressorParams:

  /** @group setParam */
  def setBaseLearner(value: Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]]): this.type = set(baseLearner, value)

  /** @group setParam */
  def setReplacement(value: Boolean): this.type = set(replacement, value)

  /** @group setParam */
  def setSampleRatio(value: Double): this.type = set(sampleRatio, value)

  /** @group setParam */
  def setReplacementFeatures(value: Boolean): this.type = set(replacementFeatures, value)

  /** @group setParam */
  def setSampleFeaturesNumber(value: Int): this.type = set(sampleFeaturesNumber, value)

  /** @group setParam */
  def setReduce(value: Array[Double] => Double): this.type = set(reduce, value)

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /**
    * Set the maximum level of parallelism to evaluate models in parallel.
    * Default is 1 for serial evaluation
    *
    * @group expertSetParam
    */
  def setParallelism(value: Int): this.type = set(parallelism, value)

  override def copy(extra: ParamMap): BaggingRegressor = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): BaggingRegressionModel = instrumented { instr =>

    //Pass some parameters automatically to baseLearner
    setBaseLearner(getBaseLearner.setFeaturesCol(getFeaturesCol).asInstanceOf[Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]]])
    setBaseLearner(getBaseLearner.setLabelCol(getLabelCol).asInstanceOf[Predictor[Vector, _ <: Predictor[Vector, _, _], _ <: PredictionModel[Vector, _]]])

    val spark = dataset.sparkSession

    instr.logPipelineStage(this)
    //    instr.logDataset(dataset)
    instr.logParams(this, maxIter, seed, parallelism)

    val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
    val toArrUdf = udf(toArr)

    val toVector: Seq[Double] => Vector = seq => Vectors.dense(seq.toArray)
    val toVectorUdf = udf(toVector)

    //TODO: Try to find spark functions for Array.fill(array_repeat), find better than if expr for withoutReplacement
    def weightBag(withReplacement: Boolean, sampleRatio: Double, numberSamples: Int, seed: Long): Column = {

      if (withReplacement) {
        array((0 until numberSamples).map(iter => poisson(sampleRatio, seed + iter)): _*)
      } else {
        if (sampleRatio == 1) {
          array_repeat(lit(1), numberSamples)
        } else {
          array((0 until numberSamples).map(iter => expr(s"if(rand($seed+$iter)<$sampleRatio,1,0)")): _*)
        }
      }

    }

    def duplicateRow(col: Column): Column = {
      explode(array_repeat(lit(1), col))
    }

    def arraySample(withReplacement: Boolean, sampleRatio: Double, seed: Long)(array: Seq[Double]): Seq[Double] = {

      if (withReplacement) {
        val poisson = new PoissonDistribution(sampleRatio)
        poisson.reseedRandomGenerator(seed)
        array.flatMap(d =>
          if (poisson.sample() > 1) {
            Seq(d)
          } else {
            Seq.empty[Double]
          }
        )
      } else {
        if (sampleRatio == 1) {
          array
        } else {
          val rng = new XORShiftRandom(seed)
          array.flatMap(d =>
            if (rng.nextDouble() < sampleRatio) {
              Seq(d)
            } else {
              Seq.empty[Double]
            }
          )
        }
      }

    }

    def arrayIndicesSample(withReplacement: Boolean, max: Int, seed: Long)(array: Array[Int]): Array[Int] = {

      if (withReplacement) {
        val rand = new Random(seed)
        Array.fill(max)(rand.nextInt(array.length)).distinct
      } else {
        if (max == array.length) {
          array
        } else {
          Random.shuffle(array.indices.toIndexedSeq).toArray.take(max)
        }
      }.sorted

    }

    def withWeightedBag(withReplacement: Boolean, sampleRatio: Double, numberSamples: Int, seed: Long, outputColName: String)(df: DataFrame): DataFrame = {
      df.withColumn(outputColName, weightBag(withReplacement, sampleRatio, numberSamples, seed))
    }

    def withSampledRows(weightsColName: String, index: Int)(df: DataFrame): DataFrame = {
      df.withColumn("dummy", duplicateRow(col(weightsColName)(index))).drop(col("dummy"))
    }

    def withSampledFeatures(featuresColName: String, indices: Array[Int])(df: DataFrame): DataFrame = {
      val slicer = udf { vec: Vector =>
        vec match {
          case features: DenseVector => Vectors.dense(indices.map(features.apply))
          case features: SparseVector => features.slice(indices)
        }
      }
      df.withColumn(featuresColName, slicer(col(featuresColName)))
    }

    def getNumFeatures(dataset: Dataset[_], maxNumFeatures: Int = 100): Int = {
      BaggingMetadataUtils.getNumFeatures(dataset.schema(getFeaturesCol)) match {
        case Some(n: Int) => n
        case None =>
          // Get number of classes from dataset itself.
          val sizeFeaturesCol: Array[Row] = dataset.select(size(col(getFeaturesCol))).take(1)
          if (sizeFeaturesCol.isEmpty || sizeFeaturesCol(0).get(0) == null) {
            throw new SparkException("ML algorithm was given empty dataset.")
          }
          val sizeArrayFeatures: Int = sizeFeaturesCol.head.getInt(0)
          val numFeatures = sizeArrayFeatures.toInt
          require(numFeatures <= maxNumFeatures, s"Classifier inferred $numFeatures from label values" +
            s" in column $labelCol, but this exceeded the max numClasses ($maxNumFeatures) allowed" +
            s" to be inferred from values.  To avoid this error for labels with > $numFeatures" +
            s" classes, specify numClasses explicitly in the metadata; this can be done by applying" +
            s" StringIndexer to the label column.")
          logInfo(this.getClass.getCanonicalName + s" inferred $numFeatures classes for" +
            s" labelCol=$labelCol since numClasses was not specified in the column metadata.")
          numFeatures
      }
    }

    val withBag = dataset.toDF().transform(withWeightedBag(getReplacement, getSampleRatio, getMaxIter, getSeed, "weightedBag"))

    val df = withBag.cache()

    val futureModels = (0 until getMaxIter).map(iter =>
      Future[IM] {

        val rowSampled = df.transform(withSampledRows("weightedBag", iter))

        val numFeatures = getNumFeatures(df, 10000)
        val featuresIndices: Array[Int] = arrayIndicesSample(getReplacementFeatures, getSampleFeaturesNumber, getSeed + iter)((0 until numFeatures).toArray)
        val rowFeatureSampled = rowSampled.transform(withSampledFeatures(getFeaturesCol, featuresIndices))

        instr.logDebug(s"Start training for $iter iteration on $rowFeatureSampled with $getBaseLearner")

        val model = getBaseLearner.fit(rowFeatureSampled)

        instr.logDebug(s"Training done for $iter iteration on $rowFeatureSampled with $getBaseLearner")

        new IM(featuresIndices, model)

      }(getExecutionContext))

    val models = futureModels.map(f => ThreadUtils.awaitResult(f, Duration.Inf))

    df.unpersist()

    new BaggingRegressionModel(models.toArray)

  }
}

//Because fuck type erasure
class IM(indices: Array[Int], model: PredictionModel[Vector, _]) extends Serializable {
  def getModel: PredictionModel[Vector, _] = model

  def getIndices: Array[Int] = indices
}

class BaggingRegressionModel(override val uid: String, models: Array[IM]) extends RegressionModel[Vector, BaggingRegressionModel] with BaggingParams {

  //def this(models: Array[(Array[Int], PredictionModel[Vector, _])]) = this(Identifiable.randomUID("BaggingRegressionModel"), models)

  //def this(models: Array[PredictionModel[Vector, _]]) = this(Array.fill(models.length)(0).map(Array(_)).zip(models))

  def this(models: Array[IM]) = this(Identifiable.randomUID("BaggingRegressionModel"), models)

  def this() = this(Array.empty[IM])

  override def predict(features: Vector): Double = getReduce(predictNormal(features))

  def predictNormal(features: Vector): Array[Double] = {
    models.map(model => {
      val indices = model.getIndices
      val subFeatures = features match {
        case features: DenseVector => Vectors.dense(indices.map(features.apply))
        case features: SparseVector => features.slice(indices)
      }
      model.getModel.predict(subFeatures)
    })
  }

  def predictFuture(features: Vector): Array[Double] = {
    val futurePredictions = models.map(model => Future[Double] {
      val indices = model.getIndices
      val subFeatures = features match {
        case features: DenseVector => Vectors.dense(indices.map(features.apply))
        case features: SparseVector => features.slice(indices)
      }
      model.getModel.predict(subFeatures)
    }(getExecutionContext))
    futurePredictions.map(ThreadUtils.awaitResult(_, Duration.Inf))
  }

  override def copy(extra: ParamMap): BaggingRegressionModel = new BaggingRegressionModel(models)

  def getModels: Array[PredictionModel[Vector, _]] = models.map(_.getModel)

}

