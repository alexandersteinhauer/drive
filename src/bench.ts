/**
 * GPU benchmark (open the game with ?bench, or run `bun run bench`): renders fixed views
 * with the physics paused, timing every GPU pass with timestamp queries, then again with
 * each part of the scene or post-processing left out, to show what each costs.
 *
 * Frames are rendered back to back (each awaited on the GPU) rather than on the display's
 * refresh, so it needs no visible window and is not capped by vsync: it runs in a background
 * tab or headless Chrome.
 */
import * as THREE from "three/webgpu";

// GPUBufferUsage / GPUMapMode flags (the globals aren't in the DOM typings here).
const QUERY_RESOLVE = 0x200;
const COPY_SRC = 0x4;
const COPY_DST = 0x8;
const MAP_READ = 0x1;

/** Set by `instrument` before the renderer exists, so every texture view gets a name. */
const viewNames = new WeakMap<GPUTextureView, string>();
const devices = new WeakMap<GPUCommandEncoder, GPUDevice>();
let timer: PassTimer | null = null;

/** Wraps WebGPU so render and compute passes can be timed and named. Call before init. */
export function instrument(): void {
  const createView = GPUTexture.prototype.createView;
  GPUTexture.prototype.createView = function (
    this: GPUTexture,
    descriptor?: GPUTextureViewDescriptor,
  ) {
    const view = createView.call(this, descriptor);
    viewNames.set(
      view,
      `${this.label || "canvas"} ${this.width}x${this.height}`,
    );
    return view;
  };
  const createEncoder = GPUDevice.prototype.createCommandEncoder;
  GPUDevice.prototype.createCommandEncoder = function (
    this: GPUDevice,
    descriptor?: GPUCommandEncoderDescriptor,
  ) {
    const encoder = createEncoder.call(this, descriptor);
    devices.set(encoder, this);
    return encoder;
  };
  const renderPass = GPUCommandEncoder.prototype.beginRenderPass;
  GPUCommandEncoder.prototype.beginRenderPass = function (
    this: GPUCommandEncoder,
    descriptor: GPURenderPassDescriptor,
  ) {
    const color = [...descriptor.colorAttachments][0];
    const depth = descriptor.depthStencilAttachment;
    const target = (color?.view ?? depth?.view) as GPUTextureView | undefined;
    const name = (target && viewNames.get(target)) ?? "canvas";
    return renderPass.call(
      this,
      timer?.wrap(
        this,
        descriptor,
        depth && !color ? `${name} (depth)` : name,
      ) ?? descriptor,
    );
  };
  const computePass = GPUCommandEncoder.prototype.beginComputePass;
  GPUCommandEncoder.prototype.beginComputePass = function (
    this: GPUCommandEncoder,
    descriptor?: GPUComputePassDescriptor,
  ) {
    const d = descriptor ?? {};
    return computePass.call(
      this,
      timer?.wrap(this, d, `compute ${d.label ?? ""}`) ?? d,
    );
  };
}

/** Timestamps around each pass of a frame; read back once the frame is done. */
class PassTimer {
  private static readonly MAX = 512;
  private readonly querySet: GPUQuerySet;
  private readonly resolve: GPUBuffer;
  private readonly read: GPUBuffer;
  private names: string[] = [];
  /** Off while measuring throughput (frames in flight, nothing read back). */
  enabled = true;

  constructor(private readonly device: GPUDevice) {
    this.querySet = device.createQuerySet({
      type: "timestamp",
      count: PassTimer.MAX,
    });
    this.resolve = device.createBuffer({
      size: PassTimer.MAX * 8,
      usage: QUERY_RESOLVE | COPY_SRC,
    });
    this.read = device.createBuffer({
      size: PassTimer.MAX * 8,
      usage: MAP_READ | COPY_DST,
    });
  }

  wrap<D extends GPURenderPassDescriptor | GPUComputePassDescriptor>(
    encoder: GPUCommandEncoder,
    descriptor: D,
    name: string,
  ): D {
    // Passes three.js times itself, other devices', and overflow go untimed.
    if (
      !this.enabled ||
      descriptor.timestampWrites ||
      devices.get(encoder) !== this.device ||
      this.names.length * 2 >= PassTimer.MAX
    )
      return descriptor;
    const i = this.names.length * 2;
    this.names.push(name);
    return {
      ...descriptor,
      timestampWrites: {
        querySet: this.querySet,
        beginningOfPassWriteIndex: i,
        endOfPassWriteIndex: i + 1,
      },
    };
  }

  /**
   * The frame's GPU busy time (ms: while any pass runs) and each pass's share of it.
   * Passes overlap on tiled GPUs (Apple's): a pass's own start-to-end includes waiting on
   * earlier ones, so each is credited with the time from the previous pass's end to its own.
   * Starts the next frame.
   */
  async collect(): Promise<{ total: number; passes: [string, number][] }> {
    const names = this.names;
    this.names = [];
    if (!names.length) return { total: 0, passes: [] };
    const encoder = this.device.createCommandEncoder();
    encoder.resolveQuerySet(
      this.querySet,
      0,
      names.length * 2,
      this.resolve,
      0,
    );
    encoder.copyBufferToBuffer(
      this.resolve,
      0,
      this.read,
      0,
      names.length * 16,
    );
    this.device.queue.submit([encoder.finish()]);
    await this.read.mapAsync(MAP_READ, 0, names.length * 16);
    const t = new BigUint64Array(
      this.read.getMappedRange(0, names.length * 16),
    );
    // Passes with no work write no timestamps (Dawn on Metal): those are skipped.
    const spans = names
      .map((name, k) => ({ name, begin: t[2 * k]!, end: t[2 * k + 1]! }))
      .filter((p) => p.begin > 0n && p.end >= p.begin)
      .sort((a, b) => (a.end < b.end ? -1 : a.end > b.end ? 1 : 0));
    this.read.unmap();
    if (!spans.length) return { total: 0, passes: [] };
    const ms = (ns: bigint) => Number(ns) / 1e6;
    let last = spans[0]!.begin;
    const passes = spans.map((p): [string, number] => {
      const from = p.begin > last ? p.begin : last;
      last = p.end;
      return [p.name, ms(p.end - from)];
    });
    // Busy time: idle gaps between passes (the GPU waiting for the CPU to submit) excluded.
    return { total: passes.reduce((sum, [, t]) => sum + t, 0), passes };
  }
}

export interface BenchContext {
  renderer: THREE.WebGPURenderer;
  scene: THREE.Scene;
  /** Renders one game frame (simulation paused). */
  frame: (now: number) => void;
  /** Puts the car at road sample `i` and builds everything around it. */
  goTo: (i: number) => Promise<void>;
  setPost: (skip: { bloom?: boolean; aa?: boolean }) => void;
  setTime: (hours: number) => void;
  /** Compiles every shader the scene now needs (after a variant changed materials). */
  compile: () => Promise<void>;
  /** Optimisations that can be switched off, to measure what they save. */
  toggles: Record<string, (on: boolean) => void>;
  parts: Record<string, THREE.Object3D[]>;
}

/** Road samples (2 m apart) to look from: open meadow, woods, hills (seed 1). */
const VIEWS = [40, 900, 2200];
const WARM = 10;
const FRAMES = 90;
/** Variants are measured in rounds alternating with the baseline, ROUND_FRAMES each. */
const ROUNDS = 6;
const ROUND_FRAMES = 30;

interface Variant {
  name: string;
  apply: () => void;
  undo: () => void;
}

/**
 * Near-tree experiments (?bench=trees): leaves and bark hidden, or drawn with cheaper
 * materials, to tell how many pixels they cover from what each pixel costs.
 */
function treeVariants(ctx: BenchContext): Variant[] {
  const meshes = ctx.parts["near trees"] as THREE.InstancedMesh[];
  const materials = new Set(
    meshes.flatMap((m) => [m.material].flat() as THREE.Material[]),
  );
  const named = (re: RegExp) => [...materials].filter((m) => re.test(m.name));
  const leaves = named(/^leaves_/);
  /** Swaps each of `list` for `make(it)` in every pool while applied. */
  const swap = (
    name: string,
    list: THREE.Material[],
    make: (m: THREE.NodeMaterial) => THREE.Material,
  ): Variant => {
    const map = new Map(list.map((m) => [m, make(m as THREE.NodeMaterial)]));
    const back = new Map([...map].map(([a, b]) => [b, a]));
    const each = (m: Map<THREE.Material, THREE.Material>) => () => {
      for (const mesh of meshes)
        mesh.material = ([mesh.material].flat() as THREE.Material[]).map(
          (x) => m.get(x) ?? x,
        );
    };
    return { name, apply: each(map), undo: each(back) };
  };
  const hide = (name: string, list: THREE.Material[]): Variant => ({
    name,
    apply: () => list.forEach((m) => (m.visible = false)),
    undo: () => list.forEach((m) => (m.visible = true)),
  });
  return [
    { name: "all", apply: () => {}, undo: () => {} },
    ...Object.entries(ctx.toggles).map(([name, set]) => ({
      name: `without ${name}`,
      apply: () => set(false),
      undo: () => set(true),
    })),
    hide("no near trees at all", [...materials]),
    hide("no leaves", leaves),
    hide("no bark", named(/^bark/)),
    hide("no rocks", named(/^rock/)),
    hide("no bushes' and plants' leaves", named(/^(plants|grass)$/)),
    swap("leaves unlit (basic)", leaves, (m) => {
      const b = new THREE.MeshBasicNodeMaterial({ side: m.side });
      b.colorNode = m.colorNode;
      b.maskNode = m.maskNode;
      return b;
    }),
    swap("leaves unlit, plain colour", leaves, (m) => {
      const b = new THREE.MeshBasicNodeMaterial({ side: m.side });
      b.maskNode = m.maskNode;
      return b;
    }),
    swap("leaves opaque (no mask)", leaves, (m) => {
      const c = m.clone() as THREE.NodeMaterial;
      c.maskNode = null;
      c.alphaTest = 0;
      return c;
    }),
    {
      name: "near: no shadow receiving",
      apply: () => meshes.forEach((m) => (m.receiveShadow = false)),
      undo: () => meshes.forEach((m) => (m.receiveShadow = true)),
    },
  ];
}

/** What is drawn of the near trees: instances and triangles per material. */
function treeCensus(ctx: BenchContext): string {
  const rows = new Map<string, { instances: number; triangles: number }>();
  for (const mesh of ctx.parts["near trees"] as THREE.InstancedMesh[]) {
    if (!mesh.count || mesh.renderOrder < 0) continue;
    const mats = [mesh.material].flat() as THREE.Material[];
    const geo = mesh.geometry;
    const groups = geo.groups.length
      ? geo.groups
      : [{ start: 0, count: geo.index!.count, materialIndex: 0 }];
    for (const g of groups) {
      const name = mats[g.materialIndex ?? 0]!.name;
      const r = rows.get(name) ?? { instances: 0, triangles: 0 };
      r.instances += mesh.count;
      r.triangles += (g.count / 3) * mesh.count;
      rows.set(name, r);
    }
  }
  return [...rows]
    .sort((a, b) => b[1].triangles - a[1].triangles)
    .map(
      ([name, r]) =>
        `  ${name.padEnd(16)} ${String(r.instances).padStart(5)} instances ${String(Math.round(r.triangles / 1000)).padStart(5)}k triangles`,
    )
    .join("\n");
}

export async function run(ctx: BenchContext, suite = ""): Promise<string> {
  const device = (ctx.renderer.backend as unknown as { device: GPUDevice })
    .device;
  timer = new PassTimer(device);
  ctx.setTime(15);
  const hide = (name: string): Variant => ({
    name: `no ${name}`,
    apply: () => ctx.parts[name]!.forEach((o) => (o.visible = false)),
    undo: () => ctx.parts[name]!.forEach((o) => (o.visible = true)),
  });
  const pixelRatio = ctx.renderer.getPixelRatio();
  const census: string[] = [];
  const variants: Variant[] =
    suite === "trees"
      ? treeVariants(ctx)
      : [
          { name: "all", apply: () => {}, undo: () => {} },
          ...Object.keys(ctx.parts).map(hide),
          {
            name: "no bloom",
            apply: () => ctx.setPost({ bloom: true }),
            undo: () => ctx.setPost({}),
          },
          {
            name: "no SMAA",
            apply: () => ctx.setPost({ aa: true }),
            undo: () => ctx.setPost({}),
          },
          {
            name: `pixel ratio ${pixelRatio * 0.75}`,
            apply: () => ctx.renderer.setPixelRatio(pixelRatio * 0.75),
            undo: () => ctx.renderer.setPixelRatio(pixelRatio),
          },
          {
            name: `pixel ratio ${pixelRatio / 2} (half)`,
            apply: () => ctx.renderer.setPixelRatio(pixelRatio / 2),
            undo: () => ctx.renderer.setPixelRatio(pixelRatio),
          },
        ];
  // Per variant: frame totals; for "all", per-pass times too.
  const totals = new Map<string, number[]>();
  const savings = new Map<string, number[]>();
  const baselines = new Map<number, number[]>();
  const spreads = new Map<string, number[]>();
  /** GPU busy time per view (timestamps), next to the throughput. */
  const busy = new Map<number, number[]>();
  const passes = new Map<string, number[]>();
  // What three.js's animation loop does before each frame: without it, node updates (the
  // scene and post passes) see the same frame and are skipped.
  const r = ctx.renderer as unknown as {
    info: { autoReset: boolean; reset(): void; frame: number };
    _nodes: { nodeFrame: { update(): void; frameId: number } };
  };
  const render = async () => {
    if (r.info.autoReset) r.info.reset();
    r._nodes.nodeFrame.update();
    r.info.frame = r._nodes.nodeFrame.frameId;
    ctx.frame(performance.now());
    await device.queue.onSubmittedWorkDone();
    return timer!.collect();
  };
  for (const at of VIEWS) {
    await ctx.goTo(at);
    for (let i = 0; i < 60; i++) await render();
    if (suite === "trees") census.push(`view ${at}:\n${treeCensus(ctx)}`);
    /**
     * GPU-bound frame time (ms): `frames` rendered back to back with two in flight, so the GPU
     * never waits (and keeps its clock up), over the wall time. The CPU's part of a frame
     * (~2 ms) is well below the GPU's, so this is the GPU's.
     */
    const throughput = async (frames: number) => {
      timer!.enabled = false;
      for (let i = 0; i < WARM; i++) await render();
      const inFlight: Promise<void>[] = [];
      const start = performance.now();
      for (let i = 0; i < frames; i++) {
        if (r.info.autoReset) r.info.reset();
        r._nodes.nodeFrame.update();
        r.info.frame = r._nodes.nodeFrame.frameId;
        ctx.frame(performance.now());
        inFlight.push(device.queue.onSubmittedWorkDone());
        if (inFlight.length > 2) await inFlight.shift();
      }
      await Promise.all(inFlight);
      const ms = (performance.now() - start) / frames;
      timer!.enabled = true;
      return ms;
    };
    /** Median frame (ms) over `frames`; per-pass medians into `into` if given. */
    const measure = async (frames: number, into?: Map<string, number[]>) => {
      for (let i = 0; i < WARM; i++) await render();
      const frameTotals: number[] = [];
      const framePasses = new Map<string, number[]>();
      for (let i = 0; i < frames; i++) {
        const { total, passes: list } = await render();
        frameTotals.push(total);
        if (!into) continue;
        // Same-named passes in a frame (bloom levels, sun field steps) are summed.
        const sums = new Map<string, number>();
        for (const [name, ms] of list)
          sums.set(name, (sums.get(name) ?? 0) + ms);
        for (const [name, ms] of sums) push(framePasses, name, ms);
      }
      if (into)
        for (const [name, l] of framePasses)
          push(into, name, median(l) * (l.length / frames));
      return median(frameTotals);
    };
    // Each variant in rounds alternating with the baseline; the saving is the median of the
    // rounds' differences, so drift (GPU clocks, background work) cancels out.
    for (const v of variants) {
      if (v.name === "all") {
        push(busy, at, await measure(FRAMES, passes));
        const base = await throughput(FRAMES);
        push(baselines, at, base);
        push(totals, "all", base);
        continue;
      }
      v.apply();
      await ctx.compile();
      v.undo();
      await ctx.compile();
      const diffs: number[] = [];
      const times: number[] = [];
      for (let r = 0; r < ROUNDS; r++) {
        const base = await throughput(ROUND_FRAMES);
        push(baselines, at, base);
        v.apply();
        const ms = await throughput(ROUND_FRAMES);
        v.undo();
        diffs.push(base - ms);
        times.push(ms);
      }
      push(totals, v.name, median(times));
      push(savings, v.name, median(diffs));
      push(spreads, v.name, Math.max(...diffs) - Math.min(...diffs));
    }
  }
  const canvas = ctx.renderer.domElement;
  const lines = [
    `GPU per frame at ${canvas.width}x${canvas.height}, mean over ${VIEWS.length} views of the per-view median (${FRAMES} frames)`,
    "(GPU busy time; passes credited with the time from the previous pass's end to their own,",
    "as they overlap on tiled GPUs. Idle gaps waiting on the CPU are left out.)",
    "",
    "passes (all on):",
    ...[...passes]
      .map(([name, l]): [string, number] => [name, mean(l)])
      .filter(([, ms]) => ms >= 0.005)
      .sort((a, b) => b[1] - a[1])
      .map(([name, ms]) => `  ${fmt(ms)}  ${name}`),
    `  ${fmt(mean([...busy.values()].flat()))}  frame (GPU busy)`,
    "",
    ...(census.length ? ["near trees drawn:", ...census, ""] : []),
    "baseline spread per view (min-max over the run): " +
      [...baselines]
        .map(
          ([at, l]) =>
            `${at}: ${Math.min(...l).toFixed(2)}-${Math.max(...l).toFixed(2)}`,
        )
        .join(", "),
    "",
    "frame times below: GPU-bound throughput (wall time per frame, frames back to back)",
    `variants (saving: median over ${ROUNDS} rounds alternating with the baseline):`,
    ...variants.map((v) => {
      const ms = mean(totals.get(v.name)!);
      if (v.name === "all")
        return `  ${fmt(ms)}  all  [views: ${totals
          .get("all")!
          .map((x) => x.toFixed(2))
          .join(" ")}]`;
      const save = savings.get(v.name)!;
      const spread = mean(spreads.get(v.name)!);
      return `  ${fmt(ms)}  ${v.name.padEnd(30)} saves ${fmt(mean(save))}  [per view: ${save.map((x) => x.toFixed(2)).join(" ")}; rounds spread ${spread.toFixed(2)}]`;
    }),
  ];
  return lines.join("\n");
}

const fmt = (ms: number) => `${ms.toFixed(2).padStart(6)} ms`;
const mean = (l: number[]) => l.reduce((a, b) => a + b, 0) / l.length;
function median(l: number[]): number {
  const s = [...l].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)]!;
}
function push<K>(map: Map<K, number[]>, key: K, value: number): void {
  const l = map.get(key);
  if (l) l.push(value);
  else map.set(key, [value]);
}
