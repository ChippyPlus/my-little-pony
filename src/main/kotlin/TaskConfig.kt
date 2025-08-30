package org.cuttlefish

import kotlinx.serialization.Serializable

@Serializable
data class TaskConfig(
    val inputBitAmount: Int, // Number of input neurons
    val outputSize: Int,     // Number of output neurons
    val epochs: Int = 5000,
    val learningRate: Double = 0.1,
    val hiddenSize: Int = 6,
    val useIntUserInputs: Boolean = false,
    val modelFileName: String,
    val trainingDataFileName: String,
    val hiddenSizesToTest: List<Int>,
    val learningRatesToTest: List<Double>,
    val dispatchers: String, // io or cpu
    val massAllModelPath: String,
    val statusFile: String
)

