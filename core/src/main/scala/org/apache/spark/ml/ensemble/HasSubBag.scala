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
import org.apache.spark.SparkException
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.bfunctions._
import org.apache.spark.sql.functions._

import java.util.UUID
import scala.util.Random

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

  setDefault(replacement -> true)

  /**
   * param for ratio of rows sampled out of the dataset
   *
   * @group param
   */
  val subsampleRatio: Param[Double] =
    new DoubleParam(
      this,
      "subsampleRatio",
      "ratio of rows sampled out of the dataset",
      ParamValidators.inRange(0, 1))

  /** @group getParam */
  def getSubsampleRatio: Double = $(subsampleRatio)

  setDefault(subsampleRatio -> 1)

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

  protected def mkSubspace(subspaceRatio: Double, numFeatures: Int, seed: Long): Array[Int] = {

    val range = Array.range(0, numFeatures)
    val rng = new Random(seed)
    range.filter(_ => rng.nextDouble() < subspaceRatio)

  }

  protected def subspace(featuresColName: String, indices: Array[Int], numFeatures: Int)(
      df: DataFrame): DataFrame = {

    // shortcut
    if (indices.sameElements(Array.range(0, numFeatures))) return df

    val tmpSubSpaceColName = "bag$tmp" + UUID.randomUUID().toString
    val vs = new VectorSlicer()
      .setInputCol(featuresColName)
      .setOutputCol(tmpSubSpaceColName)
      .setIndices(indices)

    vs.transform(df)
      .withColumn(featuresColName, col(tmpSubSpaceColName))
      .drop(tmpSubSpaceColName)

  }

  protected def bootstrap(replacement: Boolean, subsampleRatio: Double, seed: Long)(
      df: DataFrame): DataFrame = {
    // shortcut
    if (!replacement && subsampleRatio == 1.0) return df

    df.sample(replacement, subsampleRatio, seed)
  }

  protected def subbag(
      featuresColName: String,
      replacement: Boolean,
      subsampleRatio: Double,
      subspaceRatio: Double,
      numFeatures: Int,
      seed: Long)(df: DataFrame) = {

    val indices = mkSubspace(subspaceRatio, numFeatures, seed)
    (
      indices,
      df.transform(bootstrap(replacement, subsampleRatio, seed))
        .transform(subspace(featuresColName, indices, numFeatures)))
  }

  protected def slicer(indices: Array[Int]): Vector => Vector = {
    case features: DenseVector => Vectors.dense(indices.map(features.apply))
    case features: SparseVector => features.slice(indices, true)
  }

}
