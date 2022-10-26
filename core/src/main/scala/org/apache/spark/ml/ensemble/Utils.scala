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

import scala.reflect.ClassTag
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.attribute.AttributeGroup

object Utils {

  def weightedMedian[
      @specialized(Double) T: ClassTag: Ordering,
      @specialized(Double) W: ClassTag: Numeric](data: Array[T], weights: Array[W]) = {
    import Numeric.Implicits._

    val (sortedData, sortedWeights) = data.zip(weights).sortBy(_._1).unzip
    val cusumWeights = sortedWeights
      .scanLeft(0.0)(_ + _.toDouble)
      .tail
    val sumWeights = cusumWeights.last
    val median = cusumWeights
      .map(_ >= 0.5 * sumWeights)
      .indexOf(true)
    sortedData(median)
  }

  def getFeaturesMetadata(
      dataset: Dataset[_],
      featuresCol: String,
      indices: Option[Array[Int]] = None): Metadata = {

    val featuresAttr = AttributeGroup.fromStructField(dataset.schema(featuresCol))

    indices match {
      case None => featuresAttr.toMetadata()
      case Some(value) => {
        val selectedAttrs = featuresAttr.attributes.map { attrs => value.map(attrs.apply) }
        val outputAttr = selectedAttrs match {
          case Some(attrs) => new AttributeGroup("features", attrs)
          case None => new AttributeGroup("features", value.length)
        }
        outputAttr.toMetadata()
      }
    }

  }

}
