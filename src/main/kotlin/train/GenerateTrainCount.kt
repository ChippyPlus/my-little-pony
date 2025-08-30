package org.cuttlefish.train




import java.io.File
import kotlinx.serialization.json.Json
import org.cuttlefish.TaskConfig
import org.cuttlefish.TrainingData
import kotlin.math.pow

fun main(args: Array<String>) {
    val config: TaskConfig = try {
        val jsonString = File("config.json").readText()
        Json.decodeFromString(jsonString)
    } catch (e: Exception) {
        println("Error loading config.json: ${e.message}")
        println("Please ensure config.json exists and is valid. Exiting.")
        return
    }
    val bitAmount = config.inputBitAmount
    val outputSize = config.outputSize

    val numbersToTrain = (0u..(2).toDouble().pow(config.inputBitAmount).toUInt() - 1u).toList()

    val inputs = mutableListOf<List<Double>>()
    val outputs = mutableListOf<List<Double>>()

    println("Generating training data for ${numbersToTrain.size} numbers...")

    for (num in numbersToTrain) {
        val binaryString = num.toString(2).padStart(bitAmount, '0')

        // Convert the binary string to a List<Double> (0.0 or 1.0)
        val binaryInput = binaryString.map { (it.code - '0'.code).toDouble() }
        inputs.add(binaryInput)

        // Calculate the number of '1's in the binary string
        val countOfOnes = binaryString.count { it == '1' }.toUInt()

        // Convert the count of '1's to a 4-bit binary string
        // Pad to outputSize (4)
        val countBinaryString = countOfOnes.toString(2).padStart(outputSize, '0')

        // Convert the count's binary string to a List<Double>
        val targetOutput = countBinaryString.map { (it.code - '0'.code).toDouble() }
        outputs.add(targetOutput)
    }

    // Create the TrainingData object
    val trainingData = TrainingData(
        inputSize = bitAmount,
        outputSize = outputSize,
        inputs = inputs,
        outputs = outputs
    )

    // Serialize the TrainingData object to a JSON string
    val jsonString = Json.encodeToString(trainingData)

    // Define the output file name
    val outputFileName = if (args.isEmpty()) "trainingData.json" else args[0]

    // Write the JSON string to the file
    File(outputFileName).writeText(jsonString)

    println("Training data generated and saved to $outputFileName")
    println("Input size: $bitAmount, Output size: $outputSize")
    println("Generated ${inputs.size} training examples.")
}