package org.apache.spark.ml.boosting
import org.apache.commons.math3.util.FastMath
import org.apache.spark.ml.classification.{ClassificationModel, Classifier}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.shared.{HasMaxIter, HasWeightCol}
import org.apache.spark.ml.param.{DoubleParam, LongParam, Param}
import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}

private[ml] trait PredictorVectorTypeTrait {

  // scalastyle:off structural.type
  type PredictorVectorType = Predictor[Vector, E, M] forSome {
    type M <: PredictionModel[Vector, M]
    type E <: Predictor[Vector, E, M]
  }
  // scalastyle:on structural.type
}

trait BoostingParams extends PredictorParams with HasMaxIter with HasWeightCol with PredictorVectorTypeTrait {

  /**
    * param for the base learner to be stacked with boosting
    *
    * @group param
    */
  val baseLearner: Param[PredictorVectorType] =
    new Param[PredictorVectorType](
      this,
      "baseLearner",
      "base learner that will get stacked with boosting"
    )

  /** @group getParam */
  def getBaseLearner: PredictorVectorType = $(baseLearner)


  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val learningRate: Param[Double] = new DoubleParam(this, "learningRate", "learning rate for computing booster weights")

  /** @group getParam */
  def getLearningRate: Double = $(learningRate)

  setDefault(learningRate -> 0.5)

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val seed: Param[Long] = new LongParam(this, "seed", "seed for randomness")

  /** @group getParam */
  def getSeed: Long = $(seed)

  setDefault(seed -> System.nanoTime())

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val reduce: Param[Array[Double] => Double] =
    new Param(this, "reduce", "function to reduce the predictions of the models")

  /** @group getParam */
  def getReduce: Array[Double] => Double = $(reduce)

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val loss: Param[Double => Double] =
    new Param(this, "loss", "loss function, exponential by default")

  /** @group getParam */
  def getLoss: Double =>Double = $(loss)

  setDefault(loss -> { error =>
    1 - FastMath.exp(-error)
  })

  setDefault(maxIter   -> 10)

}
