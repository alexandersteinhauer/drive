import type { WebGPURenderer } from "three/webgpu";

const W = 220;
const ROW = 44;
const HISTORY = W;
const DPR = Math.min(devicePixelRatio, 2);

interface Row {
  label: string;
  color: string;
  values: Float32Array;
  text: string;
}

function makeRow(label: string, color: string): Row {
  return {
    label,
    color,
    values: new Float32Array(HISTORY).fill(NaN),
    text: `${label} -`,
  };
}

/**
 * Performance overlay: FPS, CPU and GPU frame time as scrolling graphs (toggle with F).
 *
 * A browser can't read system-wide CPU/GPU utilisation, so both are this game's own load:
 * CPU is the main-thread time spent in the frame callback (simulation, scene updates and
 * WebGPU command encoding), GPU is the time from the GPU starting the frame's work to
 * finishing it (WebGPU timestamp queries). Each is graphed as a share of the frame interval, so
 * 100% means that side is what limits the frame rate.
 */
export class Stats {
  private readonly canvas = document.createElement("canvas");
  private readonly ctx: CanvasRenderingContext2D;
  private readonly fps = makeRow("FPS", "#7dff9b");
  private readonly cpu = makeRow("CPU", "#6cc8ff");
  private readonly gpu = makeRow("GPU", "#ffb35c");
  private readonly rows: Row[];
  private readonly gpuSupported: boolean;
  private head = 0;
  private start = 0;
  private lastFrame = 0;
  private gpuMs = NaN;
  private readonly gpuTimer: GpuTimer | null;
  // Averages over the current text-refresh window.
  private sumInterval = 0;
  private sumCpu = 0;
  private sumGpu = 0;
  private gpuCount = 0;
  private count = 0;
  private lastText = 0;

  constructor(renderer: WebGPURenderer) {
    this.gpuSupported = renderer.hasFeature("timestamp-query");
    this.gpuTimer = this.gpuSupported
      ? new GpuTimer(
          (renderer.backend as unknown as { device: GPUDevice }).device,
          (ms) => (this.gpuMs = ms),
        )
      : null;
    this.rows = [this.fps, this.cpu, this.gpu];
    if (!this.gpuSupported) this.gpu.text = "GPU n/a (no timestamp-query)";
    this.canvas.id = "stats";
    this.canvas.width = W * DPR;
    this.canvas.height = ROW * this.rows.length * DPR;
    this.canvas.style.width = `${W}px`;
    this.canvas.style.height = `${ROW * this.rows.length}px`;
    this.ctx = this.canvas.getContext("2d")!;
    this.ctx.scale(DPR, DPR);
    this.visible = localStorage.getItem("stats") !== "0";
    document.body.append(this.canvas);
  }

  get visible(): boolean {
    return !this.canvas.hidden;
  }

  set visible(on: boolean) {
    this.canvas.hidden = !on;
    localStorage.setItem("stats", on ? "1" : "0");
    // Don't let a gap while hidden read as one long frame.
    this.lastFrame = 0;
  }

  /** Call first thing in the frame callback. */
  begin(): void {
    this.start = performance.now();
    if (this.visible) this.gpuTimer?.begin();
  }

  /** Call after the frame has been rendered. */
  end(): void {
    this.gpuTimer?.end();
    if (!this.visible) return;
    const now = performance.now();
    const cpu = now - this.start;
    const interval = this.lastFrame ? this.start - this.lastFrame : NaN;
    this.lastFrame = this.start;
    if (!(interval > 0)) return;

    const fps = 1000 / interval;
    this.fps.values[this.head] = fps;
    this.cpu.values[this.head] = (100 * cpu) / interval;
    this.gpu.values[this.head] = (100 * this.gpuMs) / interval;
    this.head = (this.head + 1) % HISTORY;

    this.sumInterval += interval;
    this.sumCpu += cpu;
    if (this.gpuMs === this.gpuMs) {
      this.sumGpu += this.gpuMs;
      this.gpuCount++;
    }
    this.count++;
    if (now - this.lastText > 250) {
      const avg = this.sumInterval / this.count;
      const pct = (ms: number) => `${Math.round((100 * ms) / avg)}%`;
      const cpuMs = this.sumCpu / this.count;
      this.fps.text = `FPS ${(1000 / avg).toFixed(0)}  ${avg.toFixed(1)} ms`;
      this.cpu.text = `CPU ${cpuMs.toFixed(1)} ms  ${pct(cpuMs)}`;
      if (this.gpuSupported && this.gpuCount) {
        const gpuMs = this.sumGpu / this.gpuCount;
        this.gpu.text = `GPU ${gpuMs.toFixed(1)} ms  ${pct(gpuMs)}`;
      }
      this.sumInterval = this.sumCpu = this.sumGpu = 0;
      this.count = this.gpuCount = 0;
      this.lastText = now;
    }
    this.draw();
  }

  private draw(): void {
    const { ctx } = this;
    ctx.clearRect(0, 0, W, ROW * this.rows.length);
    // FPS scale: the next multiple of 30 above the highest value shown.
    let maxFps = 60;
    for (const v of this.fps.values) if (v > maxFps) maxFps = v;
    maxFps = Math.ceil(maxFps / 30) * 30;
    this.rows.forEach((row, i) => {
      const top = i * ROW;
      const max = i === 0 ? maxFps : 100;
      ctx.fillStyle = "rgb(0 0 0 / 55%)";
      ctx.fillRect(0, top, W, ROW - 2);
      ctx.fillStyle = row.color;
      ctx.globalAlpha = 0.75;
      const graphTop = top + 16;
      const graphH = ROW - 2 - 16;
      for (let x = 0; x < HISTORY; x++) {
        const v = row.values[(this.head + x) % HISTORY] ?? NaN;
        if (v !== v) continue;
        const h = Math.max(1, (Math.min(v, max) / max) * graphH);
        ctx.fillRect(x, graphTop + graphH - h, 1, h);
      }
      ctx.globalAlpha = 1;
      ctx.font = "600 12px ui-monospace, monospace";
      ctx.fillText(row.text, 6, top + 12);
    });
  }
}

// WebGPU's GPUBufferUsage / GPUMapMode flags (not in TypeScript's DOM lib).
const MAP_READ = 0x0001;
const COPY_SRC = 0x0004;
const COPY_DST = 0x0008;
const QUERY_RESOLVE = 0x0200;
/** Query resolves must land 256-byte aligned: one 256-byte stretch per slot. */
const SLOTS = 4;

/**
 * GPU time of a frame: timestamps written by a tiny compute pass submitted before the
 * frame's work and another after it. Per-pass timestamps (three's trackTimestamp) can't be
 * summed for this: on tiled GPUs such as Apple's, passes overlap, so their durations add up
 * to several times the frame. Results arrive a few frames late; frames are skipped while
 * every slot is still waiting on its readback.
 */
class GpuTimer {
  private readonly querySet: GPUQuerySet;
  private readonly pipeline: GPUComputePipeline;
  private readonly resolve: GPUBuffer;
  private readonly read: GPUBuffer[] = [];
  private readonly busy: boolean[] = [];
  private current = -1;

  constructor(
    private readonly device: GPUDevice,
    private readonly onResult: (ms: number) => void,
  ) {
    // Passes with no work record no timestamps (Dawn on Metal), so each marker runs a no-op.
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: "@compute @workgroup_size(1) fn main() {}",
        }),
        entryPoint: "main",
      },
    });
    this.querySet = device.createQuerySet({
      type: "timestamp",
      count: 2 * SLOTS,
    });
    this.resolve = device.createBuffer({
      size: 256 * SLOTS,
      usage: QUERY_RESOLVE | COPY_SRC,
    });
    for (let i = 0; i < SLOTS; i++) {
      this.read.push(
        device.createBuffer({ size: 16, usage: MAP_READ | COPY_DST }),
      );
      this.busy.push(false);
    }
  }

  begin(): void {
    this.current = this.busy.indexOf(false);
    if (this.current < 0) return;
    this.busy[this.current] = true;
    this.mark({ beginningOfPassWriteIndex: 2 * this.current });
  }

  end(): void {
    const slot = this.current;
    this.current = -1;
    if (slot < 0) return;
    const encoder = this.mark({ endOfPassWriteIndex: 2 * slot + 1 }, false);
    encoder.resolveQuerySet(
      this.querySet,
      2 * slot,
      2,
      this.resolve,
      256 * slot,
    );
    const read = this.read[slot]!;
    encoder.copyBufferToBuffer(this.resolve, 256 * slot, read, 0, 16);
    this.device.queue.submit([encoder.finish()]);
    read
      .mapAsync(1)
      .then(() => {
        const [start, end] = new BigUint64Array(read.getMappedRange());
        read.unmap();
        if (start && end && end > start)
          this.onResult(Number(end - start) / 1e6);
      })
      .catch(() => {})
      .finally(() => (this.busy[slot] = false));
  }

  private mark(
    writes: Omit<GPUComputePassTimestampWrites, "querySet">,
    submit = true,
  ): GPUCommandEncoder {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass({
      timestampWrites: { querySet: this.querySet, ...writes },
    });
    pass.setPipeline(this.pipeline);
    pass.dispatchWorkgroups(1);
    pass.end();
    if (submit) this.device.queue.submit([encoder.finish()]);
    return encoder;
  }
}
