package org.cuttlefish

import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.Json
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.system.exitProcess

val config: TaskConfig = try {
	Json.decodeFromString(File("config.json").readText())
} catch (e: Exception) {
	println("Error loading config.json: ${e.message}")
	exitProcess(1)
}

data class TrainingResult(
		val hiddenSize: Int,
		val learningRate: Double,
		val variant: String,
		val finalError: Double,
		val epochsRun: Int,
		val model: MultiLayerPerceptron,
						 )

data class ProgressInfo(
		val hiddenSize: Int,
		val learningRate: Double,
		val variant: String,
		val totalEpochs: Int,
		var epoch: Int = 0,
		var avgError: Double = Double.MAX_VALUE,
		private var lastAvgError: Double = Double.MAX_VALUE,
		var status: String = "QUEUED",
					   ) {
	val errorImprovement: Double
		get() = if (lastAvgError != Double.MAX_VALUE) lastAvgError - avgError else 0.0

	fun updateError(newError: Double) {
		lastAvgError = avgError
		avgError = newError
	}
}

private var uniDashCount = 0

// MODIFICATION: Updated dashboard to show the new model ID format [HS-LR-V]
suspend fun redrawDashboard(mutex: Mutex, progressMap: Map<Int, ProgressInfo>, totalRuns: Int) {
	mutex.withLock {
		// Move cursor up by the total number of lines to overwrite the previous dashboard
		print("\u001B[${totalRuns}A")

		// Calculate padding for neat columns based on the config values
		val hsWidth = config.hiddenSizesToTest.maxOrNull()?.toString()?.length ?: 2
		val lrWidth = config.learningRatesToTest.maxOfOrNull { "%.3f".format(it).length } ?: 5
		val epochWidth = config.epochs.toString().length

		val consoleOutput = StringBuilder()
		val fileOutput = StringBuilder()

		// Start the console output by moving the cursor up to the beginning of the dashboard area
		consoleOutput.append("\u001B[${totalRuns}A")


		// Sort the jobs by error so the best-performing ones are always at the top
		val sortedProgress = progressMap.values.sortedBy { it.avgError }

		sortedProgress.forEachIndexed { index, info ->
			val rank = index + 1
			// Clear the current line before printing the new one
			print("\r\u001B[K")

			// Create a consistently padded model ID (e.g., "[128-0.100-A]") for alignment
			val modelId = String.format(
				"[%-${hsWidth}d-%-${lrWidth}.3f-%s]", info.hiddenSize, info.learningRate, info.variant
									   )

			val line = when (info.status) {
				"QUEUED" -> "  [%-2d] %s [QUEUED]    | Waiting to start...".format(
					rank, modelId
																				  )

				"RUNNING" -> {
					val progressPercent = (info.epoch.toDouble() / info.totalEpochs * 100).toInt()
					"  [%-2d] %s [RUNNING]   | Epoch: %-${epochWidth}d (%3d%%) | Error: %.12f | Improvement: %+.12f".format(
						rank, modelId, info.epoch, progressPercent, info.avgError, info.errorImprovement
																														   )
				}

				"COMPLETED", "CONVERGED" -> "  [%-2d] %s [%-9s] | Finished in %-${epochWidth}d epochs | Final Error: %.12f".format(
					rank, modelId, info.status, info.epoch, info.avgError
																																  )

				else -> "  [UNKNOWN] An error occurred."
			}
			consoleOutput.append("\u001B[K").append(line).append("\n")
			fileOutput.append(line).append("\n")
			println(line)
		}

		try {
			File(config.statusFile).writeText(fileOutput.toString())
		} catch (e: Exception) {
			println("Error writing to status file: ${e.message}")
		}
	}
}


fun main() = runBlocking {
	println("--- Starting Mass Training Experiment ---")

	val trainingDataList: List<Pair<List<Double>, List<Double>>> = try {
		val jsonString = File(config.trainingDataFileName).readText()
		val loadedTrainingData = Json.decodeFromString<TrainingData>(jsonString)
		loadedTrainingData.inputs.zip(loadedTrainingData.outputs)
	} catch (e: Exception) {
		println("Error loading training data from ${config.trainingDataFileName}: ${e.message}")
		return@runBlocking
	}
	println("Training data loaded with ${trainingDataList.size} samples.")

	val hiddenSizesToTest = config.hiddenSizesToTest
	val learningRatesToTest = config.learningRatesToTest

	val trainingRuns = hiddenSizesToTest.flatMap { hs ->
		learningRatesToTest.flatMap { lr ->
			listOf(Triple(hs, lr, "A"), Triple(hs, lr, "B"))
		}
	}
	val totalRuns = trainingRuns.size

	println("Will perform $totalRuns training runs in parallel.")
	println("-------------------------------------------------")

	val results = ConcurrentLinkedQueue<TrainingResult>()
	val consoleMutex = Mutex()
	val progressMap = ConcurrentHashMap<Int, ProgressInfo>()

	trainingRuns.forEachIndexed { index, (hs, lr, variant) ->
		progressMap[index] = ProgressInfo(
			hiddenSize = hs, learningRate = lr, variant = variant, totalEpochs = config.epochs
										 )
	}

	repeat(totalRuns) { println() }
	redrawDashboard(consoleMutex, progressMap, totalRuns)

	withContext(if (config.dispatchers == "io") Dispatchers.IO else Dispatchers.Default) {
		val jobs = trainingRuns.mapIndexed { index, (hiddenSize, learningRate, variant) ->
			launch {
				val jobScope = this

				val mlp = MultiLayerPerceptron(
					inputSize = config.inputBitAmount,
					hiddenSize = hiddenSize,
					outputSize = config.outputSize,
					learningRate = learningRate,
					epochs = config.epochs
											  )

				val progressCallback = { epoch: Int, _: Int, avgError: Double ->
					val info = progressMap[index]!!
					info.epoch = epoch
					info.updateError(avgError)
					if (info.status == "QUEUED") {
						info.status = "RUNNING"
					}
					jobScope.launch {
						redrawDashboard(consoleMutex, progressMap, totalRuns)
					}
				}

				val (finalError, epochsRun) = mlp.train(trainingDataList, progressCallback as Function3<Int, Int, Double, Unit>)
				results.add(TrainingResult(hiddenSize, learningRate, variant, finalError, epochsRun, mlp))

				val info = progressMap[index]!!
				info.epoch = epochsRun
				info.updateError(finalError)
				info.status = if (epochsRun < config.epochs && finalError < 0.000001) "CONVERGED" else "COMPLETED"

				redrawDashboard(consoleMutex, progressMap, totalRuns)
			}
		}
		jobs.joinAll()
	}

	println("\n-------------------------------------------------")
	println("All training runs complete. Analyzing results...")
	File(config.massAllModelPath).mkdirs()
	File(config.massAllModelPath).deleteRecursively()
	File(config.massAllModelPath).mkdirs()
	if (results.isEmpty()) {
		println("No results were generated.")
		return@runBlocking
	}

	val bestResult = results.minByOrNull { it.finalError }!!
	val bestModelId = "[${bestResult.hiddenSize}-${bestResult.learningRate}-${bestResult.variant}]"

	println("\n--- Best Performing Model ---")
	println("Model ID: $bestModelId")
	println("Epochs to Converge: ${bestResult.epochsRun}")
	println("Final Average Error: %.21f".format(bestResult.finalError))
	println("-----------------------------\n")

	// MODIFICATION: Use the standardized ID for the best model filename
	val bestModelFileName = "best_model_${bestModelId}.json"
	bestResult.model.save(bestModelFileName)
	println("Saved best model to $bestModelFileName")

	println("\n--- Full Report (sorted by error) ---")
	println("Rank | Model ID          | Final Error          | Epochs")
	println("-----|-------------------|----------------------|-------")

	// MODIFICATION: Display the standardized ID in the final report
	val sortedResults = results.sortedBy { it.finalError }
	sortedResults.forEachIndexed { index, result ->
		val modelId = "[${result.hiddenSize}-${"%.3f".format(result.learningRate)}-${result.variant}]"
		System.out.printf(
			"%-4d | %-17s | %-20.16f | %d\n", index + 1, modelId, result.finalError, result.epochsRun
						 )
	}

	// MODIFICATION: Use the standardized ID for all saved model filenames
	sortedResults.forEachIndexed { index, result ->
		val modelId = "[${result.hiddenSize}-${result.learningRate}-${result.variant}]"
		val fileName = "${config.massAllModelPath}/rank_${index + 1}_${modelId}_err_${result.finalError}.json"
		result.model.save(fileName)
	}
	println("\nSaved all models to '${config.massAllModelPath}' directory.")
	return@runBlocking Unit
}