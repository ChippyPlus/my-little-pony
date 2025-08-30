package org.cuttlefish

import java.io.File
import kotlinx.serialization.json.Json
import kotlin.math.roundToInt

 fun main() {

    val configFileName = "config.json"

    val config: TaskConfig = try {
        val jsonString = File(configFileName).readText()
        Json.decodeFromString(jsonString)
    } catch (e: Exception) {
        println("Error loading config.json: ${e.message}")
        println("Please ensure config.json exists and is valid. Exiting.")
        return
    }
    val modelFileName = config.modelFileName
    val trainingDataFileName = config.trainingDataFileName

    println("Loaded configuration: $config")

    var finalMLP: MultiLayerPerceptron

    if (File(modelFileName).exists()) {
        println("Loading existing MLP model from $modelFileName...")
        finalMLP = MultiLayerPerceptron.load(modelFileName)
        println("Model loaded successfully.")

        // Verify loaded model matches config (optional, but good for robustness)
        if (finalMLP.inputSize != config.inputBitAmount || finalMLP.outputSize != config.outputSize) {
            println("WARNING: Loaded model's input/output sizes do not match config.json!")
            println("Model: Input=${finalMLP.inputSize}, Output=${finalMLP.outputSize}")
            println("Config: Input=${config.inputBitAmount}, Output=${config.outputSize}")
            println("Consider deleting $modelFileName and $trainingDataFileName to retrain with current config.")
            // You might choose to exit here or force retraining
        }

    } else {
        val trainingDataList: List<Pair<List<Double>, List<Double>>>
        if (File(trainingDataFileName).exists()) {
            println("Loading training data from $trainingDataFileName...")
            val jsonString = File(trainingDataFileName).readText()
            val loadedTrainingData = Json.decodeFromString<TrainingData>(jsonString)
            trainingDataList = loadedTrainingData.inputs.zip(loadedTrainingData.outputs)
            println("Training data loaded successfully. Input size: ${loadedTrainingData.inputSize}, Output size: ${loadedTrainingData.outputSize}.")

            if (loadedTrainingData.inputSize != config.inputBitAmount || loadedTrainingData.outputSize != config.outputSize) {
                println("WARNING: Loaded training data's input/output sizes do not match config.json!")
                println("Training Data: Input=${loadedTrainingData.inputSize}, Output=${loadedTrainingData.outputSize}")
                println("Config: Input=${config.inputBitAmount}, Output=${config.outputSize}")
                println("Please ensure trainingData.json was generated with the current config. Running GenerateTrain.kt is recommended.")
                return
            }

        } else {
            println("No training data found ($trainingDataFileName). Please run GenerateTrain.kt first to generate training data based on your config.json.")
            return
        }

        val myMLP = MultiLayerPerceptron(
            inputSize = config.inputBitAmount,
            hiddenSize = config.hiddenSize,
            outputSize = config.outputSize,
            learningRate = config.learningRate,
            epochs = config.epochs
        )

        println("Starting training...")
        myMLP.train(trainingDataList, null)

        finalMLP = myMLP
        finalMLP.save(modelFileName)
        println("Model saved to $modelFileName")
    }

    println("\n--- Interactive Mode (Generic MLP) ---")
    println("Input bit amount: ${config.inputBitAmount}, Output size: ${config.outputSize}")
    println("Type 'e' to exit.")


    while (true) {
        val imp = readln().toInt().toString(2).padStart(config.inputBitAmount,'0')
        if (imp == "e") break

        val binaryInputForPrediction = imp.map { (it.code - '0'.code).toDouble() }
        val predictionOutputList = finalMLP.predict(binaryInputForPrediction)

        println("Input: $imp")
        println("Raw info  R: ${predictionOutputList.map { it.roundToInt() }.joinToString() }")
        println("Raw info nR: ${predictionOutputList.joinToString() }")
    }



}

fun binaryListToDecimal(binaryList: List<Double>): Int {
    var decimal = 0
    for (i in binaryList.indices) {
        val bit = if (binaryList[binaryList.size - 1 - i] > 0.5) 1 else 0 // Round to 0 or 1
        decimal += bit * (1 shl i)
    }
    return decimal
}