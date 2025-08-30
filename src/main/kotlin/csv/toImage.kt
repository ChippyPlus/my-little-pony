package org.cuttlefish.csv

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO


fun main() {
	val data = mutableListOf<Array<IntArray>>()
	val dataNumbers = mutableListOf<Int>()
	val fn = "/Users/adam/dev/kotlin/neural-network/data/mnist_test.csv"
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
				dataNumbers.add(row[0].toInt())
				data.add(dataR)
			}
		}
	}
	val numbersCountFull = mutableMapOf<Int, Int>()
	val numbersCount = mutableMapOf<Int, Int>()

	(0..9).forEach { itOut ->
		numbersCountFull[itOut] = dataNumbers.count {
			it == itOut
		}
	}

	for (imageTemplate in data.withIndex()) {
		val image = BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB)

		val imageNumber = dataNumbers[imageTemplate.index]
		numbersCount[imageNumber] = numbersCount[imageNumber]?.plus(1) ?: 1

		for (row in imageTemplate.value.withIndex()) {
			for (pixel in row.value.withIndex()) {
				with(image) { setRGB(/* x = */ pixel.index,/* y = */ row.index,/* rgb = */ pixel.value) }
			}
		}
		ImageIO.write(image, "png", File("images/N$imageNumber C${numbersCount[imageNumber]}.png"))
		println("Saved = \"images/N$imageNumber C${numbersCount[imageNumber]}\"")
	}
}

