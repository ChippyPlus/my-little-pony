package org.cuttlefish

import kotlinx.serialization.json.Json
import java.io.File

fun main() {


	val configFileName = "config.json"

	val config: TaskConfig = try {
		val jsonString = File(configFileName).readText()
		Json.decodeFromString(jsonString)
	} catch (e: Exception) {
		println("Error loading config.json: ${e.message}")
		println("Please ensure config.json exists and is valid. Exiting training.")
		return
	}

	val modelFileName = config.modelFileName
	val trainingDataFileName = config.trainingDataFileName

	println("Loaded configuration for training: $config")

	val trainingDataList: List<Pair<List<Double>, List<Double>>>
	if (File(trainingDataFileName).exists()) {
		println("Loading training data from $trainingDataFileName...")
		val jsonString = File(trainingDataFileName).readText()
		val loadedTrainingData = Json.decodeFromString<TrainingData>(jsonString)
		trainingDataList = loadedTrainingData.inputs.zip(loadedTrainingData.outputs)
		println("Training data loaded successfully. Input size: ${loadedTrainingData.inputSize}, Output size: ${loadedTrainingData.outputSize}.")

		if (loadedTrainingData.inputSize != config.inputBitAmount || loadedTrainingData.outputSize != config.outputSize) {
			println("ERROR: Loaded training data's input/output sizes do not match config.json!")
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

	myMLP.save(modelFileName)
	println("Model saved to $modelFileName")
}