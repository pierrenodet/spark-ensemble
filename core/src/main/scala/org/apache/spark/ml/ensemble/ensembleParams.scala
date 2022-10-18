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
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.ParamValidators
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util.DefaultParamsReader
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.DataFrame
import org.json4s.JObject

private[ml] trait HasNumBaseLearners extends Params {

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

private[ml] trait HasBaseLearner[L <: EnsemblePredictorType] extends Params {

  /**
   * param for the estimator that will be used by the ensemble learner as a base learner
   *
   * @group param
   */
  val baseLearner: Param[L] =
    new Param[L](this, "baseLearner", "base learner that will be used by the ensemble learner")

  /** @group getParam */
  def getBaseLearner: L = $(baseLearner)

  protected def fitBaseLearner(
      baseLearner: L,
      labelColName: String,
      featuresColName: String,
      predictionColName: String,
      weightColName: Option[String])(df: DataFrame): EnsemblePredictionModelType = {
    val paramMap = new ParamMap()
    paramMap.put(baseLearner.labelCol -> labelColName)
    paramMap.put(baseLearner.featuresCol -> featuresColName)
    paramMap.put(baseLearner.predictionCol -> predictionColName)

    if (weightColName.isDefined) {
      val baseLearner_ = baseLearner.asInstanceOf[L with HasWeightCol]
      paramMap.put(baseLearner_.weightCol -> weightColName.get)
      baseLearner_.fit(df, paramMap)
    } else {
      baseLearner.fit(df, paramMap)
    }
  }

}

private[ml] object HasBaseLearner {

  def saveImpl(
      instance: HasBaseLearner[_],
      path: String,
      sc: SparkContext,
      extraMetadata: Option[JObject] = None): Unit = {

    val learnerPath = new Path(path, "learner").toString
    instance.getBaseLearner.asInstanceOf[MLWritable].save(learnerPath)

  }

  def loadImpl[L <: EnsemblePredictorType](path: String, sc: SparkContext): L = {

    val learnerPath = new Path(path, "learner").toString
    DefaultParamsReader.loadParamsInstance[L](learnerPath, sc)

  }

}

private[ml] trait HasStacker extends Params {

  /**
   * param for the estimator that will be used by the ensemble learner to aggregate results of
   * base learner(s)
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

private[ml] object HasStacker {

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

private[ml] trait HasBaseLearners[L <: EnsemblePredictorType]
    extends HasBaseLearner[L]
    with Params {

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

}

private[ml] object HasBaseLearners {

  def saveImpl(
      instance: HasBaseLearners[_],
      path: String,
      sc: SparkContext,
      extraMetadata: Option[JObject] = None): Unit = {

    instance.getBaseLearners.map(_.asInstanceOf[MLWritable]).zipWithIndex.foreach {
      case (model, idx) =>
        val modelPath = new Path(path, s"learner-$idx").toString
        model.save(modelPath)
    }

  }

  def loadImpl[L <: EnsemblePredictorType](
      path: String,
      sc: SparkContext): Array[EnsemblePredictorType] = {

    val pathFS = new Path(path)
    val fs = pathFS.getFileSystem(sc.hadoopConfiguration)
    val modelsPath = fs.listStatus(pathFS).map(_.getPath).filter(_.getName.startsWith("learner-"))
    modelsPath.map { modelPath =>
      DefaultParamsReader.loadParamsInstance[L](modelPath.toString, sc)
    }
  }

}
