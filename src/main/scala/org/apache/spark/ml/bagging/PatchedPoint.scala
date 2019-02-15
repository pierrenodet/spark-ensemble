package org.apache.spark.ml.bagging
import breeze.linalg.DenseVector
import org.apache.commons.math3.distribution.PoissonDistribution
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

object PatchedPoint {

  def convertToPatchedRDD(inputs: RDD[Instance], indices: Array[Int]): RDD[Instance] =
    inputs.map {
      case Instance(label, weight, features) =>
        Instance(
          label,
          weight,
          Vectors.dense(features.toArray.zip(indices).flatMap{case(f,i) => if(i==0) None else Some(f)}))
    }

  def patch(
      subsamplingRate: Double,
      numFeatures: Int,
      withReplacement: Boolean,
      seed: Long = Utils.random.nextLong()): Array[Int] = {
    if (withReplacement) {
      patchSamplingWithReplacement(subsamplingRate, numFeatures, seed)
    } else {
      if (subsamplingRate == 1.0) {
        patchWithoutSampling(numFeatures)
      } else {
        patchSamplingWithoutReplacement(subsamplingRate, numFeatures, seed)
      }
    }
  }

  private def patchSamplingWithoutReplacement(
      subsamplingRate: Double,
      numFeatures: Int,
      seed: Long): Array[Int] = {
    val rng = new XORShiftRandom
    rng.setSeed(seed)
    val patch = new Array[Int](numFeatures)
    var index = 0
    while (index < numFeatures) {
      if (rng.nextDouble() < subsamplingRate) {
        patch(index) = 1
      }
      index += 1
    }
    patch
  }

  private def patchSamplingWithReplacement(
      subsample: Double,
      numFeatures: Int,
      seed: Long): Array[Int] = {
    val poisson = new PoissonDistribution(subsample)
    poisson.reseedRandomGenerator(seed)
    val patch = new Array[Int](numFeatures)
    var index = 0
    while (index < numFeatures) {
      patch(index) = if (poisson.sample() >= 1) 1 else 0
      index += 1
    }
    patch
  }

  private def patchWithoutSampling(numFeatures: Int): Array[Int] =
    Array.fill(numFeatures)(1)

}
