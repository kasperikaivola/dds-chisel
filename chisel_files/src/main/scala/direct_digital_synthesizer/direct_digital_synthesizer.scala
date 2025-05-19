// Dsp-block DirectDigitalSynthesizer
// Direct Digital Synthesizer (DDS)

package direct_digital_synthesizer

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.stage.{ChiselStage, ChiselGeneratorAnnotation}

// IO definition
class dds_io(val accWidth: Int, val outWidth: Int) extends Bundle {
  val initdone = Input(Bool())
  val A        = Input(UInt(accWidth.W))
  val B        = Output(SInt(outWidth.W))
}

/**
  * DDS module using a phase accumulator and sine LUT
  * @param accWidth Width of the phase accumulator
  * @param lutBits Number of address bits for the sine LUT
  * @param outWidth Output sample width
  * @param lutType 0 = sine, 1 = square, 2 = triangle
  */
class direct_digital_synthesizer(
  val accWidth: Int = 32, 
  val lutBits:  Int = 10, 
  val outWidth: Int = 12,
  val lutType:  Int = 0)
  extends Module {

  val io = IO(new dds_io(accWidth, outWidth))

  // Phase accumulator
  val phase = RegInit(0.U(accWidth.W))
  /*when(io.initdone) {
    phase := phase + io.A
  }*/
  phase := phase + io.A

  // Sine LUT
  /*val lutDepth = 1 << lutBits
  val sineLut = VecInit((0 until lutDepth).map { i =>
    val angle = (2.0 * math.Pi * i) / lutDepth
    val maxAmp = (1 << (outWidth - 1)) - 1
    val amp = (math.sin(angle) * maxAmp).round.toInt
    amp.S(outWidth.W)
  })
  val test = amp.S(outWidth.W)
  val idx = phase(accWidth - 1, accWidth - lutBits)
  io.B := sineLut(idx)*/

  // waveform LUT (sine / square / triangle)
  val lutDepth = 1 << lutBits
  val wavLut = VecInit((0 until lutDepth).map { i =>
    val angle  = 2.0 * math.Pi * i / lutDepth
    val maxAmp = (1 << (outWidth - 1)) - 1
    val raw = lutType match {
      case 0 => (math.sin(angle) * maxAmp).round.toInt
      case 1 => math.signum(math.sin(angle)).toInt * maxAmp
      case 2 => (2.0 * maxAmp / math.Pi * math.asin(math.sin(angle))).round.toInt
      case _ => 0
      }
      raw.S(outWidth.W)
    })
  val idx = phase(accWidth - 1, accWidth - lutBits)
  io.B := wavLut(idx)
}
  

/**
  * Verilog generator
  */
object direct_digital_synthesizer extends App {
  // 0=sine, 1=square, 2=triangle
  val lutType = System.getProperty("waveform", "0").toInt  
  val annos = Seq(ChiselGeneratorAnnotation(() =>
    new direct_digital_synthesizer(
      accWidth = 32,
      lutBits = 10,
      outWidth = 12,
      lutType = lutType
    )
  ))
  (new ChiselStage).execute(args, annos)
}
