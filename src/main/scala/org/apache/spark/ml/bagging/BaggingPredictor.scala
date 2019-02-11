package org.apache.spark.ml.bagging

import org.apache.commons.math3.distribution.PoissonDistribution
import org.apache.spark.SparkException
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.util.BaggingMetadataUtils
import org.apache.spark.sql.bfunctions.poisson
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}
import org.apache.spark.util.random.XORShiftRandom

import scala.util.Random

trait BaggingPredictor {
/*
  val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
  val toArrUdf = udf(toArr)

  val toVector: Seq[Double] => Vector = seq => Vectors.dense(seq.toArray)
  val toVectorUdf = udf(toVector)
*/
  //TODO: Try to find spark functions for Array.fill(array_repeat), find better than if expr for withoutReplacement
  def weightBag(withReplacement: Boolean, sampleRatio: Double, numberSamples: Int, seed: Long): Column = {

    require(sampleRatio > 0, "sampleRatio must be strictly positive")
    if (withReplacement) {
      array((0 until numberSamples).map(iter => poisson(sampleRatio, seed + iter)): _*)
    } else {
      if (sampleRatio == 1) {
        array_repeat(lit(1), numberSamples)
      } else {
        require(sampleRatio <= 1, s"Without replacement, the sampleRatio cannot be greater to one")
        array((0 until numberSamples).map(iter => expr(s"if(rand($seed+$iter)<$sampleRatio,1,0)")): _*)
      }
    }

  }

  def duplicateRow(col: Column): Column = {
    explode(array_repeat(lit(1), col.cast(IntegerType)))
  }

  def arraySample(withReplacement: Boolean, sampleRatio: Double, seed: Long)(array: Seq[Double]): Seq[Double] = {

    if (withReplacement) {
      val poisson = new PoissonDistribution(sampleRatio)
      poisson.reseedRandomGenerator(seed)
      array.flatMap(
        d =>
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
        array.flatMap(
          d =>
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

    val take = max.min(array.length)
    if (withReplacement) {
      val rand = new Random(seed)
      Array.fill(take)(rand.nextInt(array.length)).distinct
    } else {
      if (take == array.length) {
        array
      } else {
        Random.shuffle(array.indices.toIndexedSeq).toArray.take(take)
      }
    }.sorted

  }

  def withWeightedBag(
    withReplacement: Boolean,
    sampleRatio: Double,
    numberSamples: Int,
    seed: Long,
    outputColName: String
  )(
    df: DataFrame
  ): DataFrame = {
    df.withColumn(outputColName, weightBag(withReplacement, sampleRatio, numberSamples, seed))
  }

  def withSampledRows(weightsColName: String, index: Int)(df: DataFrame): DataFrame = {
    df.withColumn("dummy", duplicateRow(col(weightsColName)(index))).drop(col("dummy"))
  }

  def withSampledRows(weightsColName: String)(df: DataFrame): DataFrame = {
    df.withColumn("dummy", duplicateRow(col(weightsColName))).drop(col("dummy"))
  }


  def withSampledFeatures(featuresColName: String, indices: Array[Int])(df: DataFrame): DataFrame = {
    val slicer = udf { vec: Vector =>
      vec match {
        case features: DenseVector  => Vectors.dense(indices.map(features.apply))
        case features: SparseVector => features.slice(indices)
      }
    }
    df.withColumn(featuresColName, slicer(col(featuresColName)))
  }

  def getNumFeatures(dataset: Dataset[_], featuresCol: String): Int = {
    BaggingMetadataUtils.getNumFeatures(dataset.schema(featuresCol)) match {
      case Some(n: Int) => n
      case None         =>
        // Get number of classes from dataset itself.
        val sizeFeaturesCol: Array[Row] = dataset.select(size(col(featuresCol))).take(1)
        if (sizeFeaturesCol.isEmpty || sizeFeaturesCol(0).get(0) == null) {
          throw new SparkException("ML algorithm was given empty dataset.")
        }
        val sizeArrayFeatures: Int = sizeFeaturesCol.head.getInt(0)
        val numFeatures = sizeArrayFeatures.toInt
        numFeatures
    }
  }

}
