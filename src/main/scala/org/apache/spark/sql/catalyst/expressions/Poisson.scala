package org.apache.spark.sql.catalyst.expressions

import org.apache.commons.math3.distribution.PoissonDistribution
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.expressions.codegen.Block._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{AnalysisException, Column}
import org.apache.spark.util.Utils

case class Poisson(left: Expression, right: Expression)
    extends BinaryExpression
    with ExpectsInputTypes
    with Stateful
    with ExpressionWithRandomSeed {

  @transient protected var poisson: PoissonDistribution = _

  @transient protected lazy val lambda: Double = left match {
    case Literal(s, FloatType)   => s.asInstanceOf[Float].toDouble
    case Literal(s, DoubleType)  => s.asInstanceOf[Double]
    case Literal(s, IntegerType) => s.asInstanceOf[Int].toDouble
    case Literal(s, LongType)    => s.asInstanceOf[Long].toDouble
    case _                       => throw new AnalysisException(s"Input argument to $prettyName must be an float, double, integer, long or null literal.")
  }

  @transient protected lazy val seed: Long = right match {
    case Literal(s, IntegerType) => s.asInstanceOf[Int]
    case Literal(s, LongType)    => s.asInstanceOf[Long]
    case _                       => throw new AnalysisException(s"Input argument to $prettyName must be an integer, long or null literal.")
  }

  override protected def initializeInternal(partitionIndex: Int): Unit = {
    poisson = new PoissonDistribution(lambda)
    poisson.reseedRandomGenerator(seed + partitionIndex)
  }

  override def nullable: Boolean = false

  override def dataType: DataType = IntegerType

  override def inputTypes: Seq[AbstractDataType] =
    Seq(TypeCollection(FloatType, DoubleType, IntegerType, LongType), TypeCollection(IntegerType, LongType))

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

object Poisson {

  def apply(lambda: Column, seed: Column): Poisson = Poisson(lambda.expr, seed.expr)

  def apply(lambda: Double, seed: Long): Poisson = Poisson(Literal(lambda, DoubleType), Literal(seed, LongType))

  def apply(lambda: Double, seed: Column): Poisson = Poisson(Literal(lambda, DoubleType), seed.expr)

}
