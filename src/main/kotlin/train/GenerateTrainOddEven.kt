package org.cuttlefish.train

import java.io.File
import kotlinx.serialization.json.Json
import org.cuttlefish.TaskConfig
import org.cuttlefish.TrainingData
import org.cuttlefish.config
import kotlin.math.pow


val configl: TaskConfig = try {
    val jsonString = File("config.json").readText()
    Json.decodeFromString(jsonString)
} catch (e: Exception) {
    println("Error loading config.json: ${e.message}")
    println("Please ensure config.json exists and is valid. Exiting.")
    error("")
}

fun main(args:Array<String>) {
    val bitAmount = configl.inputBitAmount
    val outputSize = configl.outputSize


    val numbersToTrain = (0u..2.0.pow(bitAmount).toUInt()).toList()

    val inputs = mutableListOf<List<Double>>()
    val outputs = mutableListOf<List<Double>>()

    println("Generating training data for ${numbersToTrain.size} numbers...")

    for (num in numbersToTrain) {
        val binaryString = num.toString(2).padStart(bitAmount, '0')

        val binaryInput = binaryString.map { (it.code - '0'.code).toDouble() }
        inputs.add(binaryInput)

        val targetOutput = listOf(((num /* + 1u */) % 2u).toDouble())
        outputs.add(targetOutput)
    }

    val trainingData = TrainingData(
        inputSize = bitAmount,
        outputSize = outputSize,
        inputs = inputs,
        outputs = outputs
    )

    val jsonString = Json.encodeToString(trainingData)

    val outputFileName = if (args.isEmpty()) "trainingData.json" else args[0]

    File(outputFileName).writeText(jsonString)

    println("Training data generated and saved to $outputFileName")
    println("Input size: $bitAmount, Output size: $outputSize")
    println("Generated ${inputs.size} training examples.")
}
