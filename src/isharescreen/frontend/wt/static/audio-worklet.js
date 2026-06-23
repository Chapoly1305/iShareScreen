// Ring-buffer AudioWorklet for the WT audio path.
//
// Receives interleaved stereo Float32Array messages from the main
// thread (one per decoded Opus frame, ~5ms of samples each). Buffers
// up to ~80ms worth and feeds the audio output device 128 frames per
// `process()` call.
//
// No SharedArrayBuffer: we use plain `port.postMessage` with
// transferable Float32Arrays. At ~200 messages/sec the postMessage
// overhead is negligible, and we avoid the COOP/COEP headers SAB
// would require.

class IssAudioRing extends AudioWorkletProcessor {
  constructor() {
    super();
    // Ring sized to hold ~80ms of stereo audio @ 48kHz = 3840 stereo
    // samples = 7680 floats. Headroom against jitter without
    // adding much latency on its own.
    this._capacity = 7680;     // floats (interleaved)
    this._ring = new Float32Array(this._capacity);
    this._write = 0;
    this._read = 0;
    this._available = 0;       // floats currently buffered
    this._dropped = 0;
    this._underruns = 0;

    this.port.onmessage = (e) => {
      const data = e.data;
      if (!(data instanceof Float32Array)) return;
      const need = data.length;
      if (need > this._capacity - this._available) {
        // Drop incoming if ring full — keep newest by sliding read.
        const overflow = need - (this._capacity - this._available);
        this._read = (this._read + overflow) % this._capacity;
        this._available -= overflow;
        this._dropped++;
      }
      // Copy in (with wrap).
      const tailRoom = this._capacity - this._write;
      if (need <= tailRoom) {
        this._ring.set(data, this._write);
      } else {
        this._ring.set(data.subarray(0, tailRoom), this._write);
        this._ring.set(data.subarray(tailRoom), 0);
      }
      this._write = (this._write + need) % this._capacity;
      this._available += need;
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0];   // outputs[0] is the first output of node
    const ch0 = out[0], ch1 = out[1] || out[0];
    const quantum = ch0.length;     // 128 typically
    const need = quantum * 2;        // stereo interleaved
    if (this._available < need) {
      this._underruns++;
      // Output silence. (Float32Array initializes to 0 — but the
      // browser may reuse the buffer, so fill explicitly.)
      ch0.fill(0); if (ch1 !== ch0) ch1.fill(0);
      return true;
    }
    for (let i = 0; i < quantum; i++) {
      ch0[i] = this._ring[this._read];
      ch1[i] = this._ring[(this._read + 1) % this._capacity];
      this._read = (this._read + 2) % this._capacity;
    }
    this._available -= need;
    return true;
  }
}

registerProcessor("iss-audio-ring", IssAudioRing);
