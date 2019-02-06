package org.apache.spark.ml.util

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.sql.types.StructField

object BaggingMetadataUtils {

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
