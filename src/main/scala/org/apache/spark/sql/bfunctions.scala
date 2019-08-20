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

package org.apache.spark.sql

import org.apache.spark.sql.catalyst.expressions.{Poisson, PoissonN, Rand, Slice}
import org.apache.spark.sql.functions.{array, array_repeat, explode, expr, lit}
import org.apache.spark.sql.types.IntegerType

object bfunctions {

  def rand(col: Column) = Column(Rand(col.expr))

  def poisson(lambda: Column, seed: Column) = Column(Poisson(lambda, seed))

  def poisson(lambda: Double, seed: Column) = Column(Poisson(lambda, seed))

  def poisson(lambda: Double, seed: Long) = Column(Poisson(lambda, seed))

  def poisson_n(lambda: Column, seed: Column) = Column(PoissonN(lambda, seed))

  def poisson_n(lambda: Double, seed: Column) = Column(PoissonN(lambda, seed))

  def poisson_n(lambda: Double, seed: Long) = Column(PoissonN(lambda, seed))

  def slice(x: Column, start: Column, length: Column): Column =
    Column(Slice(x.expr, start.expr, length.expr))

  def replicate_row(col: Column): Column = {
    explode(array_repeat(lit(0), col.cast(IntegerType)))
  }

  def bag(
      withReplacement: Boolean,
      sampleRatio: Double,
      numberSamples: Int,
      seed: Long): Column = {

    require(sampleRatio > 0, "sampleRatio must be strictly positive")
    if (withReplacement) {
      array((0 until numberSamples).map(iter => poisson(sampleRatio, seed + iter)): _*)
    } else {
      if (sampleRatio == 1) {
        array_repeat(lit(1), numberSamples)
      } else {
        require(
          sampleRatio <= 1,
          s"Without replacement, the sampleRatio cannot be greater to one")
        array(
          (0 until numberSamples)
            .map(iter => expr(s"if(rand($seed+$iter)<$sampleRatio,1,0)")): _*)
      }
    }

  }

}
