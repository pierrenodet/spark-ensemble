package org.apache.spark.ml.ensemble
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.util.{DefaultParamsReader, MLWritable}
import org.apache.spark.sql.DataFrame
import org.json4s.JObject

trait HasNumRound extends Params {

  /**
   * param for the number of round waiting for next decrease in validation set
   *
   * @group param
   */
  val numRound: Param[Int] =
    new IntParam(
      this,
      "numRound",
      "number of round waiting for next decrease in validation set",
      ParamValidators.gtEq(1))

  /** @group getParam */
  def getNumRound: Int = $(numRound)

}

trait HasLearningRate extends Params {

  /**
   * param for the learning rate of the algorithm
   *
   * @group param
   */
  val learningRate: Param[Double] =
    new DoubleParam(
      this,
      "learningRate",
      "learning rate for the estimator",
      ParamValidators.gt(0.0))

  /** @group getParam */
  def getLearningRate: Double = $(learningRate)

}

trait HasNumBaseLearners extends Params {

  /**
   * param for the number of base learners of the algorithm
   *
   * @group param
   */
  val numBaseLearners: Param[Int] =
    new IntParam(
      this,
      "numBaseLearners",
      "number of base learners that will be used by the ensemble learner",
      ParamValidators.gtEq(1))

  /** @group getParam */
  def getNumBaseLearners: Int = $(numBaseLearners)

}

trait HasBaseLearner extends Params {

  /**
   * param for the estimator that will be used by the ensemble learner as a base learner
   *
   * @group param
   */
  val baseLearner: Param[EnsemblePredictorType] =
    new Param[EnsemblePredictorType](
      this,
      "baseLearner",
      "base learner that will be used by the ensemble learner")

  /** @group getParam */
  def getBaseLearner: EnsemblePredictorType = $(baseLearner)

  def fitBaseLearner(
      baseLearner: EnsemblePredictorType,
      labelColName: String,
      featuresColName: String,
      predictionColName: String,
      weightColName: Option[String])(df: DataFrame): EnsemblePredictionModelType = {
    val paramMap = new ParamMap()
    paramMap.put(baseLearner.labelCol -> labelColName)
    paramMap.put(baseLearner.featuresCol -> featuresColName)
    paramMap.put(baseLearner.predictionCol -> predictionColName)

    if (weightColName.isDefined) {
      val baseLearner_ = baseLearner.asInstanceOf[EnsemblePredictorType with HasWeightCol]
      paramMap.put(baseLearner_.weightCol -> weightColName.get)
      baseLearner_.fit(df, paramMap)
    } else {
      baseLearner.fit(df, paramMap)
    }
  }

}

object HasBaseLearner {

  def saveImpl(
      instance: HasBaseLearner,
      path: String,
      sc: SparkContext,
      extraMetadata: Option[JObject] = None): Unit = {

    val learnerPath = new Path(path, "learner").toString
    instance.getBaseLearner.asInstanceOf[MLWritable].save(learnerPath)

  }

  def loadImpl(path: String, sc: SparkContext): EnsemblePredictorType = {

    val learnerPath = new Path(path, "learner").toString
    DefaultParamsReader.loadParamsInstance[EnsemblePredictorType](learnerPath, sc)

  }

}

trait HasBaseProbabilisticClassifier extends Params {

  /**
   * param for the estimator that will be used by the ensemble learner as a base probabilistic classifier
   *
   * @group param
   */
  val baseProbabilisticClassifier: Param[EnsembleProbabilisticClassifierType] =
    new Param[EnsembleProbabilisticClassifierType](
      this,
      "baseProbabilisticClassifier",
      "base learner that will be used by the ensemble learner")

  /** @group getParam */
  def getBaseProbabilisticClassifier: EnsembleProbabilisticClassifierType =
    $(baseProbabilisticClassifier)

}

object HasBaseProbabilisticClassifier {

  def saveImpl(
      instance: HasBaseProbabilisticClassifier,
      path: String,
      sc: SparkContext,
      extraMetadata: Option[JObject] = None): Unit = {

    val learnerPath = new Path(path, "learner").toString
    instance.getBaseProbabilisticClassifier.asInstanceOf[MLWritable].save(learnerPath)

  }

  def loadImpl(path: String, sc: SparkContext): EnsembleProbabilisticClassifierType = {

    val learnerPath = new Path(path, "learner").toString
    DefaultParamsReader.loadParamsInstance[EnsembleProbabilisticClassifierType](learnerPath, sc)

  }

}

trait HasStacker extends Params {

  /**
   * param for the estimator that will be used by the ensemble learner to aggregate results of base learner(s)
   *
   * @group param
   */
  val stacker: Param[EnsemblePredictorType] =
    new Param[EnsemblePredictorType](
      this,
      "stacker",
      "stacker that will be used by the ensemble learner to aggregate results of base learner(s)")

  /** @group getParam */
  def getStacker: EnsemblePredictorType = $(stacker)

}

object HasStacker {

  def saveImpl(
      instance: HasStacker,
      path: String,
      sc: SparkContext,
      extraMetadata: Option[JObject] = None): Unit = {

    val stackerPath = new Path(path, "stacker").toString
    instance.getStacker.asInstanceOf[MLWritable].save(stackerPath)

  }

  def loadImpl(path: String, sc: SparkContext): EnsemblePredictorType = {

    val stackerPath = new Path(path, "stacker").toString
    DefaultParamsReader.loadParamsInstance[EnsemblePredictorType](stackerPath, sc)

  }

}

trait HasBaseLearners extends Params {

  /**
   * param for the estimators that will be used by the ensemble learner as base learners
   *
   * @group param
   */
  val baseLearners: Param[Array[EnsemblePredictorType]] =
    new Param[Array[EnsemblePredictorType]](
      this,
      "baseLearners",
      "base learners that will be used by the ensemble learner")

  /** @group getParam */
  def getBaseLearners: Array[EnsemblePredictorType] = $(baseLearners)

  def fitBaseLearner(
      baseLearner: EnsemblePredictorType,
      labelColName: String,
      featuresColName: String,
      predictionColName: String,
      weightColName: Option[String])(df: DataFrame): EnsemblePredictionModelType = {
    val paramMap = new ParamMap()
    paramMap.put(baseLearner.labelCol -> labelColName)
    paramMap.put(baseLearner.featuresCol -> featuresColName)
    paramMap.put(baseLearner.predictionCol -> predictionColName)

    if (weightColName.isDefined) {
      val baseLearner_ = baseLearner.asInstanceOf[EnsemblePredictorType with HasWeightCol]
      paramMap.put(baseLearner_.weightCol -> weightColName.get)
      baseLearner_.fit(df, paramMap)
    } else {
      baseLearner.fit(df, paramMap)
    }
  }

}

object HasBaseLearners {

  def saveImpl(
      instance: HasBaseLearners,
      path: String,
      sc: SparkContext,
      extraMetadata: Option[JObject] = None): Unit = {

    instance.getBaseLearners.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach {
      case (model, idx) =>
        val modelPath = new Path(path, s"learner-$idx").toString
        model.save(modelPath)
    }

  }

  def loadImpl(path: String, sc: SparkContext): Array[EnsemblePredictorType] = {

    val pathFS = new Path(path)
    val fs = pathFS.getFileSystem(sc.hadoopConfiguration)
    val modelsPath = fs.listStatus(pathFS).map(_.getPath).filter(_.getName.startsWith("learner-"))
    modelsPath.map { modelPath =>
      DefaultParamsReader.loadParamsInstance[EnsemblePredictorType](modelPath.toString, sc)
    }
  }

}
