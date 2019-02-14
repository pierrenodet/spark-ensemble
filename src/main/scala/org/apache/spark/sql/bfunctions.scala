package org.apache.spark.sql

import org.apache.spark.sql.catalyst.expressions.{Poisson, Rand, Slice}

object bfunctions {

  def rand(col: Column) = Column(Rand(col.expr))

  def poisson(lambda: Column, seed: Column) = Column(Poisson(lambda, seed))

  def poisson(lambda: Double, seed: Column) = Column(Poisson(lambda, seed))

  def poisson(lambda: Double, seed: Long) = Column(Poisson(lambda, seed))

  def slicec(x: Column, start: Column, length: Column): Column =
    Column(Slice(x.expr, start.expr, length.expr))

}
