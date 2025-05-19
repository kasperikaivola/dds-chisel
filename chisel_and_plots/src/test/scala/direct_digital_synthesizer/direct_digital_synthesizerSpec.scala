package direct_digital_synthesizer

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class direct_digital_synthesizerSpec extends AnyFlatSpec with ChiselScalatestTester {

  //behavior of "DirectDigitalSynthesizer"

  it should "generate correct amplitude for a fixed tuning word" in {
    test(new direct_digital_synthesizer) { dut =>
      // Set a tuning word that should give a stable sine output
      // e.g., 1 full cycle every 1024 clocks
      val tuningWord = (BigInt(1) << 32) / 1024

      dut.io.A.poke(tuningWord.U)

      // Step through a few cycles and print output
      for (i <- 0 until 10) {
        dut.clock.step(1)
        val amp = dut.io.B.peek().litValue
        println(s"Cycle $i: amplitude = $amp")
      }
    }
  }
}
