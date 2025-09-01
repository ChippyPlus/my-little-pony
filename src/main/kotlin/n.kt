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

    private data class UpdateResult(
            val error: Double,
            val weightsIhUpdates: Array<DoubleArray>,
            val biasesHUpdates: DoubleArray,
            val weightsHoUpdates: Array<DoubleArray>,
            val biasesOUpdates: DoubleArray
                                   ) {
		override fun equals(other: Any?): Boolean {
			if (this === other) return true
			if (javaClass != other?.javaClass) return false

			other as UpdateResult

			if (error != other.error) return false
			if (!weightsIhUpdates.contentDeepEquals(other.weightsIhUpdates)) return false
			if (!biasesHUpdates.contentEquals(other.biasesHUpdates)) return false
			if (!weightsHoUpdates.contentDeepEquals(other.weightsHoUpdates)) return false
			if (!biasesOUpdates.contentEquals(other.biasesOUpdates)) return false

			return true
		}

		override fun hashCode(): Int {
			var result = error.hashCode()
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
            epochIterationsTime: Int = 100
                     ): Pair<Double, Int> = withContext(Dispatchers.Default) {
        var lastError = 0.0
        var finalEpoch = epochs
        var averageError = 0.0

        for (epoch in 1..epochs) {
            val deferredResults = trainingData.map { (inputs, targetOutputs) ->
                async {
                    // Forward pass
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

                    var totalError = 0.0
                    for (k in 0 until outputSize) {
                        totalError += (targetOutputs[k] - outputOutputs[k]).let { it * it }
                    }

                    val outputDeltas = DoubleArray(outputSize) { k ->
                        (targetOutputs[k] - outputOutputs[k]) * sigmoidDerivative(outputOutputs[k])
                    }

                    val hiddenDeltas = DoubleArray(hiddenSize)
                    for (j in 0 until hiddenSize) {
                        var sum = 0.0
                        for (k in 0 until outputSize) {
                            sum += outputDeltas[k] * weightsHo[j][k]
                        }
                        hiddenDeltas[j] = sum * sigmoidDerivative(hiddenOutputs[j])
                    }

                    val weightsHoUpdates = Array(hiddenSize) { j ->
                        DoubleArray(outputSize) { k -> learningRate * outputDeltas[k] * hiddenOutputs[j] }
                    }
                    val biasesOUpdates = DoubleArray(outputSize) { k -> learningRate * outputDeltas[k] }

                    val weightsIhUpdates = Array(inputSize) { i ->
                        DoubleArray(hiddenSize) { j -> learningRate * hiddenDeltas[j] * inputs[i] }
                    }
                    val biasesHUpdates = DoubleArray(hiddenSize) { j -> learningRate * hiddenDeltas[j] }

                    UpdateResult(totalError, weightsIhUpdates, biasesHUpdates, weightsHoUpdates, biasesOUpdates)
                }
            }

            val results = deferredResults.awaitAll()

            var totalEpochError = 0.0
            results.forEach { result ->
                totalEpochError += result.error

                for (k in 0 until outputSize) {
                    biasesO[k] += result.biasesOUpdates[k]
                    for (j in 0 until hiddenSize) {
                        weightsHo[j][k] += result.weightsHoUpdates[j][k]
                    }
                }

                for (j in 0 until hiddenSize) {
                    biasesH[j] += result.biasesHUpdates[j]
                    for (i in 0 until inputSize) {
                        weightsIh[i][j] += result.weightsIhUpdates[i][j]
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
            lastError = averageError
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