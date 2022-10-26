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

import org.scalacheck.Arbitrary
import org.scalacheck.Gen
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class UtilsSuite extends AnyFunSuite with BeforeAndAfterAll with ScalaCheckPropertyChecks {

  import org.scalacheck.Shrink.shrinkAny

  test("weighted median with uniform weights is the unweigheted median") {

    forAll(Gen.nonEmptyListOf(Arbitrary.arbitrary[Int])) { case data =>
      val wm = Utils.weightedMedian(data.toArray, Array.fill(data.size)(1.0))
      val median = data.sorted.apply((data.size - 1) / 2)
      assert(wm == median)
    }
  }

  test(
    "weighted median with weight to zero equivalent to removing a sample for unweighted median") {

    val gen = for {
      n <- Gen.chooseNum(10, 100)
      data <- Gen.listOfN(n, Arbitrary.arbitrary[Int])
      weights <- Gen.listOfN(n, Gen.oneOf(Seq(0.0, 1.0)))
    } yield (data, weights)

    forAll(gen) { case (data, weights) =>
      val wm = Utils.weightedMedian(data.toArray, weights.toArray)
      val filtered = data.zip(weights).filter(_._2 != 0.0).map(_._1)
      val median = filtered.sorted.apply((filtered.size - 1) / 2)
      assert(wm == median)
    }
  }

  test("weighted median doesn't change when weights are scaled") {

    val gen = for {
      data <- Gen.nonEmptyListOf(Arbitrary.arbitrary[Int])
      scale <- Gen.chooseNum(0.1f, 100f)
    } yield (data, scale)

    forAll(gen) { case (data, scale) =>
      val wm = Utils.weightedMedian(data.toArray, Array.fill(data.size)(1.0))
      val wmScaled = Utils.weightedMedian(data.toArray, Array.fill(data.size)(scale))
      assert(wm == wmScaled)
    }
  }

}
