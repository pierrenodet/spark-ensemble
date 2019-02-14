package org.apache.spark.ml.stacking
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.shared.HasParallelism
import org.apache.spark.ml.param.{LongParam, Param, ParamPair, PredictorVectorTypeTrait}
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter, MLWritable}
import org.json4s.JObject
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

trait StackingParams extends PredictorParams with HasParallelism with PredictorVectorTypeTrait {

  /**
   * param for the base learner to be stacked with boosting
   *
   * @group param
   */
  val learners: Param[Array[PredictorVectorType]] =
    new Param[Array[PredictorVectorType]](this, "learners", "learners that will get stacked")

  /** @group getParam */
  def getLearners: Array[PredictorVectorType] = $(learners)

  /**
   * param for the base learner to be stacked with boosting
   *
   * @group param
   */
  val stacker: Param[PredictorVectorType] =
    new Param[PredictorVectorType](this, "stacker", "learner that will stack all the learners")

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

object StackingParams extends PredictorVectorTypeTrait {

  def saveImpl(
      path: String,
      instance: StackingParams,
      sc: SparkContext,
      extraMetadata: Option[JObject] = None): Unit = {

    val params = instance.extractParamMap().toSeq
    val jsonParams = render(
      params
        .filter { case ParamPair(p, v) => p.name != "learners" && p.name != "stacker" }
        .map { case ParamPair(p, v) => p.name -> parse(p.jsonEncode(v)) }
        .toList)

    DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, Some(jsonParams))

    instance.getLearners.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach {
      case (learner, idx) =>
        val learnerPath = new Path(path, s"learner-$idx").toString
        learner.save(learnerPath)
    }
    val stackerPath = new Path(path, s"stacker").toString
    instance.getStacker.asInstanceOf[MLWritable].save(stackerPath)

  }

  def loadImpl(path: String, sc: SparkContext, expectedClassName: String)
    : (DefaultParamsReader.Metadata, Array[PredictorVectorType], PredictorVectorType) = {

    val metadata = DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
    val pathFS = new Path(path)
    val fs = pathFS.getFileSystem(sc.hadoopConfiguration)
    val learnersPath = fs
      .listStatus(pathFS)
      .map(_.getPath)
      .filter(_.getName.startsWith("learner-"))
      .map(_.toString)
    val learners =
      learnersPath.map(DefaultParamsReader.loadParamsInstance[PredictorVectorType](_, sc))
    val stackerPath = new Path(path, "stacker").toString
    val stacker = DefaultParamsReader.loadParamsInstance[PredictorVectorType](stackerPath, sc)
    (metadata, learners, stacker)
  }

}
