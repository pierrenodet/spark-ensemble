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

package org.apache.spark.ml.bagging

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.ensemble.EnsemblePredictorType
import org.apache.spark.ml.ensemble.HasBaseLearner
import org.apache.spark.ml.ensemble.HasNumBaseLearners
import org.apache.spark.ml.ensemble.HasSubBag
import org.apache.spark.ml.param.shared.HasParallelism
import org.apache.spark.ml.param.shared.HasWeightCol

private[ml] trait BaggingParams[L <: EnsemblePredictorType]
    extends PredictorParams
    with HasNumBaseLearners
    with HasParallelism
    with HasWeightCol
    with HasBaseLearner[L]
    with HasSubBag {

  setDefault(numBaseLearners -> 10)

}
