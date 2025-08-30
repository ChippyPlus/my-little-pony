package org.cuttlefish

import kotlinx.serialization.Serializable

@Serializable
data class MLPState(
    val inputSize: Int,
    val hiddenSize: Int,
    val outputSize: Int,
    val weightsIh: List<List<Double>>,
    val biasesH: List<Double>,
    val weightsHo: List<List<Double>>,
    val biasesO: List<Double>
)