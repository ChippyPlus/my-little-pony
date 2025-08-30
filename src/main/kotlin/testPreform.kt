package org.cuttlefish

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import kotlin.math.roundToInt
import kotlinx.serialization.decodeFromString

// NOTE: I'm assuming your `TrainingData` and `MultiLayerPerceptron` classes are defined elsewhere.

/**
 * A new data class to hold both the binary vector and its decimal representation.
 */
@Serializable
data class BinaryData(
    val binary: List<Double>,
    val decimal: Int
)

/**
 * Represents the evaluation result for a single input.
 * Now uses the BinaryData class for more descriptive reports.
 */
@Serializable
data class EvaluationResult(
    val input: BinaryData,
    val predictedOutput: BinaryData,
    val expectedOutput: BinaryData,
    val isCorrect: Boolean
)

/**
 * A comprehensive report for a single model's performance on the entire dataset.
 */
@Serializable
data class ModelReport(
    val modelName: String,
    val totalCorrect: Int,
    val totalTested: Int,
    val accuracy: Double,
    val results: List<EvaluationResult>
)

/**
 * Helper function to convert a list of doubles representing binary digits
 * into its integer decimal equivalent.
 */
fun binaryToDecimal(binaryList: List<Double>): Int {
    // Converts [1.0, 0.0, 1.0] -> "101"
    val binaryString = binaryList.map { it.roundToInt() }.joinToString("")
    // Parses "101" -> 5. Returns 0 if the string is not a valid binary number.
    return binaryString.toIntOrNull(2) ?: 0
}

fun main() {
    // Use a Json instance with pretty printing for readable output files
    val json = Json {}

    // Define source and destination directories
    val modelsDir = File("allModels")
    val outputDir = File("allTrainResults")

    // Create the output directory if it doesn't exist
    outputDir.mkdirs()

    // Load the test data once
    val trainingData = Json.decodeFromString<TrainingData>(File("trainingData.json").readText())
    val totalTestCases = trainingData.inputs.size

    // Safely get a list of model files, ignoring any non-model or metadata files
    val modelFiles = modelsDir.listFiles { file ->
        file.isFile && file.name.startsWith("model_") && file.extension == "json"
    } ?: return // Exit if the directory doesn't exist or is not a directory

    println("Found ${modelFiles.size} models to evaluate.")

    // Process each model file
    for (modelFile in modelFiles) {
        val modelName = modelFile.nameWithoutExtension.removePrefix("model_")
        println("\n--- Evaluating model: $modelName ---")

        val mlp: MultiLayerPerceptron = MultiLayerPerceptron.load(modelFile.path)

        val evaluationResults = mutableListOf<EvaluationResult>()
        var amountCorrect = 0

        // Iterate through all test cases
        trainingData.inputs.zip(trainingData.outputs).forEachIndexed { index, (input, expectedOutput) ->
            // Call predict() only once per test case
            val prediction = mlp.predict(input)
            val roundedPrediction = prediction.map { it.roundToInt().toDouble() }

            val isCorrect = roundedPrediction == expectedOutput
            if (isCorrect) {
                amountCorrect++
                println("Test case ${index + 1}/$totalTestCases: ✅")
            } else {
                println("Test case ${index + 1}/$totalTestCases: ❌")
            }

            // Create the detailed EvaluationResult object with the new nested structure
            evaluationResults.add(
                EvaluationResult(
                    input = BinaryData(
                        binary = input,
                        decimal = binaryToDecimal(input)
                    ),
                    predictedOutput = BinaryData(
                        binary = roundedPrediction,
                        decimal = binaryToDecimal(roundedPrediction)
                    ),
                    expectedOutput = BinaryData(
                        binary = expectedOutput,
                        decimal = binaryToDecimal(expectedOutput)
                    ),
                    isCorrect = isCorrect
                )
            )
        }

        // After all test cases for this model are done, create the final report
        val report = ModelReport(
            modelName = modelName,
            totalCorrect = amountCorrect,
            totalTested = totalTestCases,
            accuracy = if (totalTestCases > 0) amountCorrect.toDouble() / totalTestCases else 0.0,
            results = evaluationResults
        )

        // Serialize the report to a JSON string
        val reportJson = json.encodeToString(report)

        // Save the report to its own file
        val reportFile = File(outputDir, "${modelName}_report.json")
        reportFile.writeText(reportJson)

        println("--- Model '$modelName' evaluation complete. Accuracy: $amountCorrect/$totalTestCases ---")
        println("Report saved to: ${reportFile.path}")
    }

    println("\nAll model evaluations are complete.")
}