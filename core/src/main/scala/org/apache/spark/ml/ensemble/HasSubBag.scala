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

package org.apache.spark.ml.ensemble
import java.util.UUID

import org.apache.spark.SparkException
import org.apache.spark.ml.ensemble.HasSubBag.SubSpace
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.util.BaggingMetadataUtils
import org.apache.spark.sql.bfunctions._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.util.random.XORShiftRandom

private[ml] trait HasSubBag extends Params with HasSeed {

  /**
   * param for whether samples are drawn with replacement
   *
   * @group param
   */
  val replacement: Param[Boolean] =
    new BooleanParam(this, "replacement", "whether samples are drawn with replacement")

  /** @group getParam */
  def getReplacement: Boolean = $(replacement)

  setDefault(replacement -> false)

  /**
   * param for ratio of rows sampled out of the dataset
   *
   * @group param
   */
  val sampleRatio: Param[Double] =
    new DoubleParam(
      this,
      "sampleRatio",
      "ratio of rows sampled out of the dataset",
      ParamValidators.inRange(0, 1))

  /** @group getParam */
  def getSampleRatio: Double = $(sampleRatio)

  setDefault(sampleRatio -> 1)

  /**
   * param for ratio of rows sampled out of the dataset
   *
   * @group param
   */
  val subspaceRatio: Param[Double] =
    new DoubleParam(
      this,
      "subspaceRatio",
      "ratio of features sampled out of the dataset",
      ParamValidators.inRange(0, 1))

  /** @group getParam */
  def getSubspaceRatio: Double = $(subspaceRatio)

  setDefault(subspaceRatio -> 1)

  def withBag(
      withReplacement: Boolean,
      sampleRatio: Double,
      numberSamples: Int,
      seed: Long,
      bagColName: String)(df: DataFrame): DataFrame = {
    df.withColumn(bagColName, bag(withReplacement, sampleRatio, numberSamples, seed))
  }

  def mkSubspace(sampleRatio: Double, numFeatures: Int, seed: Long): SubSpace = {

    val range = Array.range(0, numFeatures)

    if (sampleRatio == 1) {
      range
    } else {
      val rng = new XORShiftRandom(seed)
      range.flatMap(i =>
        if (rng.nextDouble() < sampleRatio) {
          Some(i)
        } else {
          None
        })
    }

  }

  def extractSubBag(bagColName: String, index: Int, featuresColName: String, subspace: SubSpace)(
      df: DataFrame): DataFrame = {

    val tmpColName = "bag$tmp" + UUID.randomUUID().toString
    val replicated = df
      .withColumn(tmpColName, replicate_row(element_at(col(bagColName), index + 1)))
      .drop(col(tmpColName))

    val tmpSubSpaceColName = "bag$tmp" + UUID.randomUUID().toString
    val vs = new VectorSlicer()
      .setInputCol(featuresColName)
      .setOutputCol(tmpSubSpaceColName)
      .setIndices(subspace)

    vs.transform(replicated)
      .withColumn(featuresColName, col(tmpSubSpaceColName))
      .drop(tmpSubSpaceColName)

  }

  def slicer(subspace: SubSpace): Vector => Vector = {
    case features: DenseVector => Vectors.dense(subspace.map(features.apply))
    case features: SparseVector => features.slice(subspace)
  }

  def getNumFeatures(dataset: DataFrame, featuresCol: String): Int = {
    BaggingMetadataUtils.getNumFeatures(dataset.schema(featuresCol)) match {
      case Some(n: Int) => n
      case None =>
        // Get number of classes from dataset itself.
        val numFeaturesUDF = udf((features: Vector) => features.size)
        val sizeFeaturesCol: Array[Row] = dataset.select(numFeaturesUDF(col(featuresCol))).take(1)
        if (sizeFeaturesCol.isEmpty || sizeFeaturesCol(0).get(0) == null) {
          throw new SparkException("ML algorithm was given empty dataset.")
        }
        val sizeArrayFeatures: Int = sizeFeaturesCol.head.getInt(0)
        val numFeatures = sizeArrayFeatures.toInt
        numFeatures
    }
  }

}

private[ml] object HasSubBag {

  type SubSpace = Array[Int]

}
