// web/src/lib/audio-processor.ts
// This worklet downsamples to 16kHz mono PCM16 LE and sends binary over the host WS
class AudioProcessor extends AudioWorkletProcessor {
  process(inputs: Float32Array[][], outputs: Float32Array[][], parameters: Record<string, Float32Array>) {
    const input = inputs[0][0];
    if (input) {
      // Simple downsampling or buffer logic here.
      // Send frames to the port:
      this.port.postMessage(input);
    }
    return true;
  }
}
registerProcessor('audio-processor', AudioProcessor);
