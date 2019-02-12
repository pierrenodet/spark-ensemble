package org.apache.spark.ml.stacking
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.shared.HasParallelism
import org.apache.spark.ml.param.{LongParam, Param, PredictorVectorTypeTrait}

trait StackingParams extends PredictorParams with HasParallelism with PredictorVectorTypeTrait {

  /**
    * param for the base learner to be stacked with boosting
    *
    * @group param
    */
  val learners: Param[Array[PredictorVectorType]] =
    new Param[Array[PredictorVectorType]](
      this,
      "learners",
      "learners that will get stacked"
    )

  /** @group getParam */
  def getLearners: Array[PredictorVectorType] = $(learners)

  /**
    * param for the base learner to be stacked with boosting
    *
    * @group param
    */
  val stacker: Param[PredictorVectorType] =
    new Param[PredictorVectorType](
      this,
      "stackers",
      "learner that will stack all the learners"
    )

  /** @group getParam */
  def getStacker: PredictorVectorType = $(stacker)

  /**
    * param for ratio of rows sampled out of the dataset
    *
    * @group param
    */
  val seed: Param[Long] = new LongParam(this, "seed", "seed for randomness")

  /** @group getParam */
  def getSeed: Long = $(seed)

  setDefault(seed -> System.nanoTime())

}
