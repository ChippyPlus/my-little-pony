package org.cuttlefish.csv

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import kotlinx.coroutines.runBlocking
import org.cuttlefish.MultiLayerPerceptron
import kotlin.time.measureTime


private const val epochT = 30
private val mlp = MultiLayerPerceptron(
	inputSize = 784, hiddenSize = 32, outputSize = 1, learningRate = 0.001, epochs = epochT
									  )
private val averageTime = LongArray(epochT)
fun main() {
	val dataFlat = mutableListOf<Array<IntArray>>()
	val data = mutableListOf<IntArray>()
	val dataNumbers = mutableListOf<Int>()
	val fn = "/Users/adam/dev/kotlin/neural-network/data/mnist_train.csv"

	csvReader().open(fn) {
		readAllAsSequence().forEachIndexed { indexOut, row: List<String> ->
			if (indexOut != 0) {
				val dataR = Array(28) { IntArray(28) }
				val rowPix = row.subList(1, row.size)
				var rowMark = -1
				rowPix.forEachIndexed { indexIn, pixelIndi ->
					if (indexIn % 28 == 0) {
						rowMark += 1
					}
					dataR[rowMark][indexIn - 28 * rowMark] = pixelIndi.toInt()
				}
				data.add(rowPix.map { it.toInt() }.toIntArray())
				dataNumbers.add(row[0].toInt())
				dataFlat.add(dataR)
			}
		}
	}
	val trainingData: MutableList<Pair<List<Double>, List<Double>>> = mutableListOf()

	for (row in data.withIndex()) {
		trainingData.add(Pair(row.value.map { it.toDouble() }, listOf(dataNumbers[row.index].toDouble())))
	}

	val x = measureTime {
		runBlocking {
			mlp.train(trainingData, progressCallback, 1)
		}
	}
	println("Time = $x(${x.inWholeMilliseconds})\nAverage Per Epoch = ${averageTime.sum() / (epochT - 1 /* cuz the 1st is 0 */)}ms")

	mlp.save("NumbersModel.json")

}

private var lastElapsedTime = 0L
private var lastTime = 0L
private var lastAvgError = 0.0
private var lastErrorChange = 0.0
private val progressCallback = { epoch: Int, _: Int, avgError: Double ->
	val currentTime = System.currentTimeMillis()
	val elapsedTime = if (lastTime != 0L) currentTime - lastTime else 0L

	val errorChange = if (lastAvgError != 0.0) lastAvgError - avgError else 0.0

	println(
		"[ E%-7d/$epochT %.5f%% ] [ A%.16f%% ] [ Time: %.5fs ] [ Error Change: %.20f ] [ E Imprv ${if (errorChange < lastErrorChange) "✅" else "❌"} ] [ Improvement ${if (avgError < lastAvgError) "✅" else "❌"} ] [ Time use ${if (avgError < lastAvgError && elapsedTime < lastElapsedTime) "✅" else "❌"} ]".format(
			epoch, epoch.toDouble() / epochT * 100, avgError, elapsedTime.toDouble() / 1000, errorChange)
		   )
	lastElapsedTime = elapsedTime
	lastTime = currentTime
	lastAvgError = avgError
	lastErrorChange = errorChange
	averageTime[epoch - 1] = elapsedTime
	// I fucking waited three hours to only realise I forgot to minus one oh my goodness its 2 am and I'm tired I just
	// did so wrong today dont worry you wrote about today its on 2/9
}


// start 20.2432061955366580%
