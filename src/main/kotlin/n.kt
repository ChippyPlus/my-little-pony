package org.cuttlefish

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import java.io.File
import kotlin.math.exp
import kotlin.random.Random

class MultiLayerPerceptron(
		val inputSize: Int,
		val hiddenSize: Int,
		val outputSize: Int,
		var learningRate: Double = 0.1,
		var epochs: Int = 1000,
		weightsIh: Array<DoubleArray>? = null,
		biasesH: DoubleArray? = null,
		weightsHo: Array<DoubleArray>? = null,
		biasesO: DoubleArray? = null
						  ) {

	// ... (All other functions like constructor, sigmoid, predict, save, load remain the same)
	val weightsIh: Array<DoubleArray> = weightsIh ?: Array(inputSize) {
		DoubleArray(hiddenSize) { Random.nextDouble(-1.0, 1.0) }
	}
	val biasesH: DoubleArray = biasesH ?: DoubleArray(hiddenSize) { Random.nextDouble(-1.0, 1.0) }

	val weightsHo: Array<DoubleArray> = weightsHo ?: Array(hiddenSize) {
		DoubleArray(outputSize) { Random.nextDouble(-1.0, 1.0) }
	}
	val biasesO: DoubleArray = biasesO ?: DoubleArray(outputSize) { Random.nextDouble(-1.0, 1.0) }

	private fun sigmoid(x: Double): Double {
		return 1.0 / (1.0 + exp(-x))
	}

	private fun sigmoidDerivative(x: Double): Double {
		return x * (1.0 - x)
	}

	fun predict(inputs: List<Double>): List<Double> {
		if (inputs.size != inputSize) {
			throw IllegalArgumentException("Input list size (${inputs.size}) must match inputSize ($inputSize).")
		}

		val hiddenOutputs = DoubleArray(hiddenSize) { 0.0 }
		for (j in 0 until hiddenSize) {
			var sum = biasesH[j] // Start with bias
			for (i in 0 until inputSize) {
				sum += inputs[i] * weightsIh[i][j]
			}
			hiddenOutputs[j] = sigmoid(sum)
		}

		val outputOutputs = DoubleArray(outputSize) { 0.0 }
		for (k in 0 until outputSize) {
			var sum = biasesO[k]
			for (j in 0 until hiddenSize) {
				sum += hiddenOutputs[j] * weightsHo[j][k]
			}
			outputOutputs[k] = sigmoid(sum)
		}
		return outputOutputs.toList()
	}


	private data class BatchUpdateResult(
			val totalError: Double,
			val weightsIhUpdates: Array<DoubleArray>,
			val biasesHUpdates: DoubleArray,
			val weightsHoUpdates: Array<DoubleArray>,
			val biasesOUpdates: DoubleArray
										) {
		override fun equals(other: Any?): Boolean {
			if (this === other) return true
			if (javaClass != other?.javaClass) return false

			other as BatchUpdateResult

			if (totalError != other.totalError) return false
			if (!weightsIhUpdates.contentDeepEquals(other.weightsIhUpdates)) return false
			if (!biasesHUpdates.contentEquals(other.biasesHUpdates)) return false
			if (!weightsHoUpdates.contentDeepEquals(other.weightsHoUpdates)) return false
			if (!biasesOUpdates.contentEquals(other.biasesOUpdates)) return false

			return true
		}

		override fun hashCode(): Int {
			var result = totalError.hashCode()
			result = 31 * result + weightsIhUpdates.contentDeepHashCode()
			result = 31 * result + biasesHUpdates.contentHashCode()
			result = 31 * result + weightsHoUpdates.contentDeepHashCode()
			result = 31 * result + biasesOUpdates.contentHashCode()
			return result
		}
	}
	suspend fun train(
			trainingData: List<Pair<List<Double>, List<Double>>>,
			progressCallback: ((Int, Int, Double) -> Unit)?,
			epochIterationsTime: Int = 100,
			batchSize: Int = 64 // Added batchSize for memory control
					 ): Pair<Double, Int> = withContext(Dispatchers.Default) {
		var finalEpoch = epochs
		var averageError = 0.0

		for (epoch in 1..epochs) {
			val batches = trainingData.chunked(batchSize)

			val deferredBatchResults = batches.map { batch ->
				async {
					val batchWeightsIhUpdates = Array(inputSize) { DoubleArray(hiddenSize) }
					val batchBiasesHUpdates = DoubleArray(hiddenSize)
					val batchWeightsHoUpdates = Array(hiddenSize) { DoubleArray(outputSize) }
					val batchBiasesOUpdates = DoubleArray(outputSize)
					var batchTotalError = 0.0

					batch.forEach { (inputs, targetOutputs) ->
						val hiddenOutputs = DoubleArray(hiddenSize)
						for (j in 0 until hiddenSize) {
							var sum = biasesH[j]
							for (i in 0 until inputSize) {
								sum += inputs[i] * weightsIh[i][j]
							}
							hiddenOutputs[j] = sigmoid(sum)
						}

						val outputOutputs = DoubleArray(outputSize)
						for (k in 0 until outputSize) {
							var sum = biasesO[k]
							for (j in 0 until hiddenSize) {
								sum += hiddenOutputs[j] * weightsHo[j][k]
							}
							outputOutputs[k] = sigmoid(sum)
						}

						for (k in 0 until outputSize) {
							batchTotalError += (targetOutputs[k] - outputOutputs[k]).let { it * it }
						}

						val outputDeltas = DoubleArray(outputSize) { k ->
							(targetOutputs[k] - outputOutputs[k]) * sigmoidDerivative(outputOutputs[k])
						}

						val hiddenDeltas = DoubleArray(hiddenSize) { j ->
							var errorSum = 0.0
							for (k in 0 until outputSize) {
								errorSum += outputDeltas[k] * weightsHo[j][k]
							}
							errorSum * sigmoidDerivative(hiddenOutputs[j])
						}

						for (k in 0 until outputSize) {
							batchBiasesOUpdates[k] += learningRate * outputDeltas[k]
							for (j in 0 until hiddenSize) {
								batchWeightsHoUpdates[j][k] += learningRate * outputDeltas[k] * hiddenOutputs[j]
							}
						}

						for (j in 0 until hiddenSize) {
							batchBiasesHUpdates[j] += learningRate * hiddenDeltas[j]
							for (i in 0 until inputSize) {
								batchWeightsIhUpdates[i][j] += learningRate * hiddenDeltas[j] * inputs[i]
							}
						}
					}
					BatchUpdateResult(batchTotalError, batchWeightsIhUpdates, batchBiasesHUpdates, batchWeightsHoUpdates, batchBiasesOUpdates)
				}
			}

			val batchResults = deferredBatchResults.awaitAll()

			var totalEpochError = 0.0
			batchResults.forEach { result ->
				totalEpochError += result.totalError

				for (k in 0 until outputSize) {
					biasesO[k] += result.biasesOUpdates[k] / batchSize
					for (j in 0 until hiddenSize) {
						weightsHo[j][k] += result.weightsHoUpdates[j][k] / batchSize
					}
				}
				for (j in 0 until hiddenSize) {
					biasesH[j] += result.biasesHUpdates[j] / batchSize
					for (i in 0 until inputSize) {
						weightsIh[i][j] += result.weightsIhUpdates[i][j] / batchSize
					}
				}
			}

			averageError = totalEpochError / trainingData.size

			if (progressCallback != null && (epoch % epochIterationsTime == 0 || epoch == 1 || epoch == epochs)) {
				progressCallback.invoke(epoch, epochs, averageError)
			}

			if (averageError < 0.00000001) {
				progressCallback?.invoke(epoch, epochs, averageError)
				finalEpoch = epoch
				break
			}
		}
		return@withContext Pair(averageError, finalEpoch)
	}

	fun save(fileName: String) {
		val mlpState = MLPState(
			inputSize = inputSize,
			hiddenSize = hiddenSize,
			outputSize = outputSize,
			weightsIh = weightsIh.map { it.toList() },
			biasesH = biasesH.toList(),
			weightsHo = weightsHo.map { it.toList() },
			biasesO = biasesO.toList()
							   )
		val jsonString = Json.encodeToString(mlpState)
		File(fileName).writeText(jsonString)
		println("MLP saved to $fileName")
	}

	companion object {
		fun load(fileName: String): MultiLayerPerceptron {
			val jsonString = File(fileName).readText()
			val mlpState = Json.decodeFromString<MLPState>(jsonString)

			return MultiLayerPerceptron(
				inputSize = mlpState.inputSize,
				hiddenSize = mlpState.hiddenSize,
				outputSize = mlpState.outputSize,
				weightsIh = mlpState.weightsIh.map { it.toDoubleArray() }.toTypedArray(),
				biasesH = mlpState.biasesH.toDoubleArray(),
				weightsHo = mlpState.weightsHo.map { it.toDoubleArray() }.toTypedArray(),
				biasesO = mlpState.biasesO.toDoubleArray()
									   )
		}
	}
}