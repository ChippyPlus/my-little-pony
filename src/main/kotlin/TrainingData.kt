package org.cuttlefish

import kotlinx.serialization.Serializable

@Serializable
data class TrainingData(
    val inputSize: Int,
    val outputSize: Int,
    val inputs: List<List<Double>>,
    val outputs: List<List<Double>>
)
