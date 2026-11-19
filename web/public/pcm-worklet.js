/*
 * AfriLingua PCM capture worklet.
 *
 * Runs on the audio rendering thread. Takes the microphone's native-rate float
 * input (usually 44.1 or 48 kHz), mixes it to mono, downsamples to 16 kHz with
 * a box-filter (averaging) resampler, converts to PCM16 little-endian and posts
 * ~40 ms frames back to the main thread as transferable ArrayBuffers.
 *
 * Messages to main thread:
 *   { type: "pcm",   buffer: ArrayBuffer }   // Int16 LE mono @ 16 kHz
 *   { type: "level", rms: number, peak: number }
 * Messages from main thread:
 *   { type: "flush" }  // emit whatever is buffered
 */
const TARGET_RATE = 16000;
const FRAME_SAMPLES = 640; // 40 ms at 16 kHz

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // `sampleRate` is a global in AudioWorkletGlobalScope.
    this.ratio = sampleRate / TARGET_RATE;
    this.acc = 0; // running sum for current output sample
    this.accCount = 0; // input samples summed so far
    this.pos = 0; // fractional progress toward next output sample
    this.out = new Int16Array(FRAME_SAMPLES);
    this.outIdx = 0;
    this.levelSum = 0;
    this.levelPeak = 0;
    this.levelCount = 0;
    this.port.onmessage = (e) => {
      if (e.data && e.data.type === "flush") this.flush();
    };
  }

  flush() {
    if (this.outIdx === 0) return;
    const chunk = this.out.slice(0, this.outIdx);
    this.port.postMessage({ type: "pcm", buffer: chunk.buffer }, [chunk.buffer]);
    this.outIdx = 0;
  }

  pushSample(v) {
    const clamped = v > 1 ? 1 : v < -1 ? -1 : v;
    this.out[this.outIdx++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    const a = Math.abs(clamped);
    this.levelSum += clamped * clamped;
    if (a > this.levelPeak) this.levelPeak = a;
    this.levelCount++;
    if (this.outIdx === FRAME_SAMPLES) {
      const chunk = this.out;
      this.port.postMessage({ type: "pcm", buffer: chunk.buffer }, [chunk.buffer]);
      this.out = new Int16Array(FRAME_SAMPLES);
      this.outIdx = 0;
      const rms = Math.sqrt(this.levelSum / Math.max(1, this.levelCount));
      this.port.postMessage({ type: "level", rms, peak: this.levelPeak });
      this.levelSum = 0;
      this.levelPeak = 0;
      this.levelCount = 0;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channels = input.length;
    const frames = input[0].length;
    for (let i = 0; i < frames; i++) {
      let s = 0;
      for (let c = 0; c < channels; c++) s += input[c][i];
      s /= channels;
      this.acc += s;
      this.accCount++;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pushSample(this.acc / this.accCount);
        this.pos -= this.ratio;
        this.acc = 0;
        this.accCount = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
