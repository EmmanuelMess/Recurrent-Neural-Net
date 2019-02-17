import org.ejml.simple.SimpleMatrix
import java.util.*
import kotlin.collections.ArrayList
import kotlin.concurrent.thread
import kotlin.math.exp

fun assert(a: Boolean) {
    if(!a) throw IllegalStateException()
}

fun doubleArr(length: Int, elem: Double) = DoubleArray(length, { elem })

fun Double.mult(x: SimpleMatrix): SimpleMatrix {
    return x.elementApply { e -> this * e }
}

class NNData(val layers: List<Int>) {
    val Os: Array<SimpleMatrix>
    var Ds: Array<SimpleMatrix>
    lateinit var e: SimpleMatrix

    init {
        Os = Array(layers.size) {
            emptySimpleMatrix()
        }

        Ds = Array(layers.size-1) { i ->
            emptySimpleMatrix()
        }
    }
}

const val c = 1
fun s(z: Double) = 1/(1 + exp(-c*z))

const val gamma: Double = 1.0

class NN(val layers: List<Int>) {
    val deltaWs: Array<SimpleMatrix>
    val extendedWs: Array<SimpleMatrix>

    val nets: MutableList<NNData>

    init {
        val rand = Random()

        extendedWs = Array(layers.size-1) { i ->
            SimpleMatrix.random_DDRM(layers[i]+1, layers[i+1], 0.0, 1.0 - Double.MIN_VALUE, rand)
        }

        deltaWs = Array(extendedWs.size) { i ->
            extendedWs[i].createLike()
        }

        nets = ArrayList()
    }

    fun process(x: DoubleArray): Double {
        var lastResult = SimpleMatrix(1, 1, true, doubleArrayOf(
            0.0
        ))

        lateinit var oTemp: SimpleMatrix

        for((timeStep, input) in x.withIndex()) {
            oTemp = SimpleMatrix(3, 1, true, doubleArrayOf(
                timeStep.toDouble(),
                lastResult[0],
                input
            ))

            for (i in 0 until extendedWs.size) {
                oTemp = oTemp.extend().rowWiseMult(extendedWs[i]).columnWiseSum().transpose().elementApply(::s)
            }

            lastResult = oTemp.copy()
        }

        return lastResult[0]
    }

    fun processForTraining(x: DoubleArray, t: DoubleArray): Double {
        nets.clear()

        var lastResult = SimpleMatrix(1, 1, true, doubleArrayOf(
            0.0
        ))

        lateinit var oTemp: SimpleMatrix

        for((timeStep, input) in x.withIndex()) {
            oTemp = SimpleMatrix(3, 1, true, doubleArrayOf(
                timeStep.toDouble(),
                lastResult[0],
                input
            ))

            val net = NNData(layers)
            nets.add(net)

            net.Os[0] = oTemp

            for (i in 0 until extendedWs.size) {
                oTemp = oTemp.extend().rowWiseMult(extendedWs[i]).columnWiseSum().transpose().elementApply(::s)

                net.Ds[i] = SimpleMatrix(net.layers[i + 1], 1, true, DoubleArray(net.layers[i + 1]) { i ->
                    ({ oi: Double -> oi * (1 - oi) })(oTemp[i, 0])
                })

                net.Os[i + 1] = oTemp
            }

            net.e = oTemp.minus(t[timeStep])

            lastResult = oTemp.copy()
        }

        return lastResult[0]
    }

    fun backpropagate() {
        var delta = SimpleMatrix(layers.last(), 1, true, doubleArr(layers.last(), 0.0))

        for((timeStep, net) in nets.withIndex().reversed()) {
            delta = net.Ds.last().mult(net.e.plus(delta))
            var deltaInner = delta

            for (i in (extendedWs.size - 1) downTo 0) {
                val deltaW = (-gamma).mult(deltaInner).mult(net.Os[i].extend().transpose()).transpose()

                if (i > 0) {
                    deltaInner = net.Ds[i - 1].rowWiseMult(extendedWs[i].unextend().matrixVectorMult(deltaInner))
                }

                deltaWs[i] = deltaWs[i].plus(deltaW)
            }


        }
    }

    fun correct(): Pair<Double, Double> {
        val error = getError()

        for (i in 0 until extendedWs.size) {
            extendedWs[i] = extendedWs[i].plus(deltaWs[i])
        }

        for(deltaW in deltaWs) {
            for(i in 0 until deltaW.numRows()) {
                for (j in 0 until deltaW.numCols()) {
                    deltaW[i, j] = 0.0
                }
            }
        }

        return Pair(error, getError())
    }

    fun getError() = nets.last().e.elementApply { (it * it)/2.0 }.elementSum()
}

fun getResult(net: NN, x: DoubleArray, y: DoubleArray): Double {
    return net.processForTraining(x, y)
}

fun trainFor(net: NN, list: List<Pair<DoubleArray, DoubleArray>>) {
    val printResults = {
        for ((input, result) in list) {
            val r = getResult(net, input, result)
            println("(${Arrays.toString(input)}) = ${r} (should be ${result.last()})")
        }
    }

    printResults()

    var iterations: Long = 0
    for (i in 0 .. Long.MAX_VALUE) {
        for((input, result) in list) {
            getResult(net, input, result)
            net.backpropagate()
        }

        net.correct()

        var errorSum = 0.0

        for((input, result) in list) {
            getResult(net, input, result)

            errorSum += net.getError()
        }

        iterations = i

        if(errorSum < 0.0001) break
        //println("${i}: $errorSum")
    }

    println("[Trained for $iterations iterations]")
    printResults()
}

const val DEBUG = false

fun main() {
    val layers = listOf(
        3,//timeStep + lastOutput1 + input
        3,
        1
    )

    val net = NN(layers)

    trainFor(
        net, listOf(
            doubleArrayOf(0.0, 0.0) to doubleArrayOf(0.0, 0.0),
            doubleArrayOf(1.0, 0.0) to doubleArrayOf(1.0, 1.0),
            doubleArrayOf(0.0, 1.0) to doubleArrayOf(0.0, 1.0),
            doubleArrayOf(1.0, 1.0, 0.0) to doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(1.0, 1.0, 1.0) to doubleArrayOf(1.0, 0.0, 1.0)
        )
    )

    println("Testing training:")

    val p = { o: DoubleArray, t: Double ->
        println("${Arrays.toString(o)} is ${net.process(o)} (should be $t)")
    }

    p(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0), 0.0)
    p(doubleArrayOf(0.0, 1.0, 1.0, 0.0, 0.0), 0.0)
    p(doubleArrayOf(1.0, 1.0, 1.0, 1.0, 0.0), 0.0)
    p(doubleArrayOf(1.0, 0.0, 0.0, 0.0, 0.0), 1.0)
    p(doubleArrayOf(0.0, 0.0, 0.0, 1.0, 0.0), 1.0)
    p(doubleArrayOf(0.0, 0.0, 0.0, 1.0, 1.0), 0.0)
}