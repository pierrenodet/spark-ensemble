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

import org.apache.spark._
import org.apache.spark.ml.ensemble.HasSubBag
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.sql._
import org.scalacheck.Arbitrary
import org.scalacheck.Gen
import org.scalacheck.Shrink
import org.scalatest.BeforeAndAfterAll
import org.scalatest.PrivateMethodTester
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._
import org.scalatest.matchers.should.Matchers._
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

case object FakeSubBag extends HasSubBag {

  override val uid: String = "FakeSubBag"

  override def copy(extra: ParamMap): Params = this

  override def subbag(
      featuresColName: String,
      replacement: Boolean,
      subsampleRatio: Double,
      subspaceRatio: Double,
      numFeatures: Int,
      seed: Long)(df: DataFrame): (Array[Int], Dataset[Row]) =
    super.subbag(featuresColName, replacement, subsampleRatio, subspaceRatio, numFeatures, seed)(
      df)

}

class HasSubBagSuite extends AnyFunSuite with BeforeAndAfterAll with ScalaCheckPropertyChecks {

  import org.scalacheck.Shrink.shrinkAny

  var spark: SparkSession = _

  override def beforeAll() {

    spark = SparkSession
      .builder()
      .config(
        new SparkConf()
          .setMaster("local[*]")
          .setAppName("example"))
      .getOrCreate()

  }

  override def afterAll() {
    spark.stop()
  }

  val featuresCol = "features"

  test("no subsampling without replacement and no subspacing is an idempotent operation") {

    val data = spark.read.format("libsvm").load("data/vehicle/vehicle.svm")
    val numFeatures = MetadataUtils.getNumFeatures(data, featuresCol)

    forAll(Gen.long) { seed =>
      val (_, subbagged) = FakeSubBag.subbag(featuresCol, false, 1, 1, numFeatures, 1L)(data)

      assert(data.collect() === subbagged.collect())
    }

  }

  test("no subspacing keeps the same amount of features") {

    val gen = for {
      replacement <- Arbitrary.arbitrary[Boolean]
      subsampleRatio <- Gen.chooseNum(0.1, 1)
      seed <- Gen.long

    } yield (replacement, subsampleRatio, seed)

    val data = spark.read.format("libsvm").load("data/vehicle/vehicle.svm")
    val numFeatures = MetadataUtils.getNumFeatures(data, featuresCol)

    forAll(gen) { case (replacement, subsampleRatio, seed) =>
      val (_, subbagged) =
        FakeSubBag.subbag("featire", replacement, subsampleRatio, 1, numFeatures, seed)(data)
      val numFeaturesSubbagged = MetadataUtils.getNumFeatures(subbagged, featuresCol)

      assert(numFeatures === numFeaturesSubbagged)
    }

  }

  test("sampling rows with replacement create duplicates") {

    val gen = for {
      seed <- Gen.long
    } yield seed

    val data = spark.read.format("libsvm").load("data/vehicle/vehicle.svm")
    val numFeatures = MetadataUtils.getNumFeatures(data, featuresCol)

    forAll(gen) { case seed =>
      val (_, subbagged) =
        FakeSubBag.subbag(featuresCol, true, 1, 1, numFeatures, seed)(data)

      assert(subbagged.count() > subbagged.distinct().count())
    }

  }

  test("subsampling ratio works") {

    val gen = for {
      subsampleRatio <- Gen.chooseNum(0.1, 1)
      seed <- Gen.long
    } yield (subsampleRatio, seed)

    val data = spark.read.format("libsvm").load("data/vehicle/vehicle.svm")
    val numFeatures = MetadataUtils.getNumFeatures(data, featuresCol)

    forAll(gen) { case (subsampleRatio, seed) =>
      val (_, subbagged) =
        FakeSubBag.subbag(featuresCol, false, subsampleRatio, 1, numFeatures, seed)(data)

      assert(data.count() * subsampleRatio === subbagged.count().toDouble +- 1e-1 * data.count())
    }

  }

  test("subspace ratio works") {

    val gen = for {
      subspaceRatio <- Gen.chooseNum(0.3, 1)
      seed <- Gen.long
    } yield (subspaceRatio, seed)

    val data = spark.read.format("libsvm").load("data/vehicle/vehicle.svm")
    val numFeatures = MetadataUtils.getNumFeatures(data, featuresCol)

    forAll(gen) { case (subspaceRatio, seed) =>
      val (_, subbagged) =
        FakeSubBag.subbag(featuresCol, true, 1, subspaceRatio, numFeatures, seed)(data)
      val numFeaturesSubbagged = MetadataUtils.getNumFeatures(subbagged, featuresCol)

      assert(numFeatures * subspaceRatio === numFeaturesSubbagged.toDouble +- 2e-1 * numFeatures)
    }

  }

}
