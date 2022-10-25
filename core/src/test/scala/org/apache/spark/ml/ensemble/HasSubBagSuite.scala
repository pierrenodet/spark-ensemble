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
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.Params
import org.apache.spark.sql._
import org.scalacheck.Gen
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class HasSubBagSuite
    extends AnyFunSuite
    with BeforeAndAfterAll
    with ScalaCheckPropertyChecks
    with HasSubBag {

  override val uid: String = "FakeSubBag"

  override def copy(extra: ParamMap): Params = this

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
    spark.sparkContext.setLogLevel("ERROR")

  }

  override def afterAll() {
    spark.stop()
  }

  test("subspace ratio works") {

    val gen = for {
      subspaceRatio <- Gen.chooseNum(0.3, 1)
      numFeatures <- Gen.chooseNum(1, 10)
      seed <- Gen.long
    } yield (subspaceRatio, numFeatures, seed)

    forAll(gen) { case (subspaceRatio, numFeatures, seed) =>
      assert(
        numFeatures * subspaceRatio ===
          subspace(subspaceRatio, numFeatures, seed).size.toDouble +- 2.5)
    }

  }

  test("subspace indices are sorted") {

    val gen = for {
      subspaceRatio <- Gen.chooseNum(0.3, 1)
      numFeatures <- Gen.chooseNum(1, 10)
      seed <- Gen.long
    } yield (subspaceRatio, numFeatures, seed)

    forAll(gen) { case (subspaceRatio, numFeatures, seed) =>
      val indices = subspace(subspaceRatio, numFeatures, seed)
      assert(indices === indices.sorted)
    }

  }

  test("no subspace keeps the same amount of features") {

    val gen = for {
      numFeatures <- Gen.chooseNum(1, 10)
      seed <- Gen.long
    } yield (numFeatures, seed)

    forAll(gen) { case (numFeatures, seed) =>
      val indices = subspace(1.0, numFeatures, seed)
      assert(indices === Array.range(0, numFeatures))
    }

  }

}
