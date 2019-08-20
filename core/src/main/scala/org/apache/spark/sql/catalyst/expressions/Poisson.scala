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

package org.apache.spark.sql.catalyst.expressions

import org.apache.commons.math3.distribution.PoissonDistribution
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.codegen.Block._
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{AnalysisException, Column}
import org.apache.spark.util.Utils

private[spark] case class Poisson(left: Expression, right: Expression)
    extends BinaryExpression
    with ExpectsInputTypes
    with Stateful
    with ExpressionWithRandomSeed {

  @transient protected var poisson: PoissonDistribution = _

  @transient protected lazy val lambda: Double = left match {
    case Literal(s, FloatType) => s.asInstanceOf[Float].toDouble
    case Literal(s, DoubleType) => s.asInstanceOf[Double]
    case Literal(s, IntegerType) => s.asInstanceOf[Int].toDouble
    case Literal(s, LongType) => s.asInstanceOf[Long].toDouble
    case _ =>
      throw new AnalysisException(
        s"Input argument to $prettyName must be an float, double, integer, long or null literal.")
  }

  @transient protected lazy val seed: Long = right match {
    case Literal(s, IntegerType) => s.asInstanceOf[Int]
    case Literal(s, LongType) => s.asInstanceOf[Long]
    case _ =>
      throw new AnalysisException(
        s"Input argument to $prettyName must be an integer, long or null literal.")
  }

  override protected def initializeInternal(partitionIndex: Int): Unit = {
    poisson = new PoissonDistribution(lambda)
    poisson.reseedRandomGenerator(seed + partitionIndex)
  }

  override def nullable: Boolean = false

  override def dataType: DataType = IntegerType

  override def inputTypes: Seq[AbstractDataType] =
    Seq(
      TypeCollection(FloatType, DoubleType, IntegerType, LongType),
      TypeCollection(IntegerType, LongType))

  def this(param: Expression) = this(param, Literal(Utils.random.nextLong(), LongType))

  def this() = this(Literal(1, DoubleType))

  override def withNewSeed(seed: Long): Poisson = Poisson(left, Literal(seed, LongType))

  override protected def evalInternal(input: InternalRow): Int = poisson.sample()

  override def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    val className = classOf[PoissonDistribution].getName
    val poissonTerm = ctx.addMutableState(className, "poisson")
    ctx.addPartitionInitializationStatement(s"""$poissonTerm = new $className($lambda);
         $poissonTerm.reseedRandomGenerator(${seed}L + partitionIndex);""".stripMargin)
    ev.copy(code = code"""final ${CodeGenerator
      .javaType(dataType)} ${ev.value} = $poissonTerm.sample();""", isNull = FalseLiteral)
  }

  override def freshCopy(): Poisson = Poisson(left, right)
}

private[spark] object Poisson {

  def apply(lambda: Column, seed: Column): Poisson = Poisson(lambda.expr, seed.expr)

  def apply(lambda: Double, seed: Long): Poisson =
    Poisson(Literal(lambda, DoubleType), Literal(seed, LongType))

  def apply(lambda: Double, seed: Column): Poisson =
    Poisson(Literal(lambda, DoubleType), seed.expr)

}

private[spark] case class PoissonN(left: Expression, right: Expression)
    extends BinaryExpression
    with ExpectsInputTypes {

  override def nullable: Boolean = false

  override def dataType: DataType = IntegerType

  override def inputTypes: Seq[AbstractDataType] =
    Seq(
      TypeCollection(FloatType, DoubleType, IntegerType, LongType),
      TypeCollection(IntegerType, LongType))

  def this(param: Expression) = this(param, Literal(Utils.random.nextLong(), LongType))

  def this() = this(Literal(1, DoubleType))

  override def eval(input: InternalRow): Any = {
    val lambda = left.eval(input) match {
      case l: Float => l.toDouble
      case l: Double => l
      case l: Int => l.toDouble
      case l: Long => l.toDouble
    }
    val seed = right.eval(input) match {
      case s: Int => s.toLong
      case s: Long => s
    }
    val poisson = new PoissonDistribution(lambda)
    poisson.reseedRandomGenerator(seed)
    poisson.sample()
  }

  override def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    val leftGen = left.genCode(ctx)
    val rightGen = right.genCode(ctx)

    val lambda = ctx.freshName("lambda")
    val seed = ctx.freshName("seed")

    val className = classOf[PoissonDistribution].getName

    val poissonTerm = ctx.addMutableState(className, "poisson")

    ev.copy(
      code"""${leftGen.code}
            |double $lambda = ${leftGen.value};
            |${rightGen.code}
            |long $seed = ${rightGen.value};
            |$poissonTerm = new $className($lambda);
            |$poissonTerm.reseedRandomGenerator($seed);
            |final double ${ev.value} = $poissonTerm.sample();""",
      isNull = FalseLiteral)
  }

}

private[spark] object PoissonN {

  def apply(lambda: Column, seed: Column): PoissonN = PoissonN(lambda.expr, seed.expr)

  def apply(lambda: Double, seed: Long): PoissonN =
    PoissonN(Literal(lambda, DoubleType), Literal(seed, LongType))

  def apply(lambda: Double, seed: Column): PoissonN =
    PoissonN(Literal(lambda, DoubleType), seed.expr)

}
