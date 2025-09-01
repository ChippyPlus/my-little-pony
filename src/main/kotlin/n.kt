package org.cuttlefish

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
		biasesO: DoubleArray? = null,
						  ) {
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

	fun train(
			trainingData: List<Pair<List<Double>, List<Double>>>,
			progressCallback: ((Int, Int, Double) -> Unit)?,
			epochIterationsTime: Int = 100,
			 ): Pair<Double, Int> {
		var lastError = 0.0
		var finalEpoch = epochs
		var averageError = 0.0

		for (epoch in 1..epochs) {
			var totalError = 0.0

			for ((inputs, targetOutputs) in trainingData) {
				val currentInputs = inputs
				val currentTargetOutputs = targetOutputs

				val hiddenOutputs = DoubleArray(hiddenSize) { 0.0 }
				val hiddenSums = DoubleArray(hiddenSize) { 0.0 }
				for (j in 0 until hiddenSize) {
					var sum = biasesH[j]
					for (i in 0 until inputSize) {
						sum += currentInputs[i] * weightsIh[i][j]
					}
					hiddenSums[j] = sum
					hiddenOutputs[j] = sigmoid(sum)
				}

				val outputOutputs = DoubleArray(outputSize) { 0.0 }
				val outputSums = DoubleArray(outputSize) { 0.0 }
				for (k in 0 until outputSize) {
					var sum = biasesO[k]
					for (j in 0 until hiddenSize) {
						sum += hiddenOutputs[j] * weightsHo[j][k]
					}
					outputSums[k] = sum
					outputOutputs[k] = sigmoid(sum)
				}

				for (k in 0 until outputSize) {
					totalError += (currentTargetOutputs[k] - outputOutputs[k]) * (currentTargetOutputs[k] - outputOutputs[k]) // Use currentTargetOutputs
				}

				val outputErrors = DoubleArray(outputSize) { 0.0 }
				val outputDeltas = DoubleArray(outputSize) { 0.0 }
				for (k in 0 until outputSize) {
					outputErrors[k] = currentTargetOutputs[k] - outputOutputs[k] // Use currentTargetOutputs
					outputDeltas[k] = outputErrors[k] * sigmoidDerivative(outputOutputs[k])
				}

				val hiddenErrors = DoubleArray(hiddenSize) { 0.0 }
				val hiddenDeltas = DoubleArray(hiddenSize) { 0.0 }
				for (j in 0 until hiddenSize) {
					var sum = 0.0
					for (k in 0 until outputSize) {
						sum += outputDeltas[k] * weightsHo[j][k]
					}
					hiddenErrors[j] = sum
					hiddenDeltas[j] = hiddenErrors[j] * sigmoidDerivative(hiddenOutputs[j])
				}

				for (k in 0 until outputSize) {
					biasesO[k] += learningRate * outputDeltas[k]
					for (j in 0 until hiddenSize) {
						weightsHo[j][k] += learningRate * outputDeltas[k] * hiddenOutputs[j]
					}
				}

				for (j in 0 until hiddenSize) {
					biasesH[j] += learningRate * hiddenDeltas[j]
					for (i in 0 until inputSize) {
						weightsIh[i][j] += learningRate * hiddenDeltas[j] * currentInputs[i] // gebruik currentInputs
					}
				}
			}


			averageError = totalError / trainingData.size

			if (progressCallback != null && (epoch % epochIterationsTime == 0 || epoch == 1 || epoch == epochs)) {
				progressCallback.invoke(epoch, epochs, averageError)
			}

			if (averageError < 0.00000001) {
				progressCallback?.invoke(epoch, epochs, averageError)
				finalEpoch = epoch
				break
			}
			lastError = averageError
		}
		return Pair(averageError, finalEpoch)
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