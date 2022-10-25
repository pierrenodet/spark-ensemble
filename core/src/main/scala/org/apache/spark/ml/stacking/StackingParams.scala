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

package org.apache.spark.ml.stacking
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.ensemble.{HasBaseLearners, HasStacker, EnsemblePredictorType}
import org.apache.spark.ml.param.shared.{HasParallelism, HasWeightCol}

private[ml] trait StackingParams[L <: EnsemblePredictorType]
    extends PredictorParams
    with HasParallelism
    with HasWeightCol
    with HasStacker[L]
    with HasBaseLearners[L] {}
