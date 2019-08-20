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

package org.apache.spark.ml.util

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.sql.types.StructField

private[spark] object BaggingMetadataUtils {

  def getNumFeatures(featuresSchema: StructField): Option[Int] = {
    val metadata = AttributeGroup.fromStructField(featuresSchema)
    if (metadata.attributes.isEmpty) {
      None
    } else {
      Some(metadata.attributes.get.toList.flatMap { attr =>
        if (attr == null) {
          None
        } else {
          Some(1)
        }
      }.sum)
    }
  }

}
