import { DefaultLoadingManager } from "three/webgpu";

/** Sizes of the files in public/ (path -> bytes), from vite.config.ts. */
declare const __PUBLIC_SIZES__: Record<string, number>;

/** Bytes received so far per file, for loaders that report progress (glTF). */
const received = new Map<string, number>();
/** Files requested through three's loaders, and those finished. */
const started = new Set<string>();
const finished = new Set<string>();
let failed: string | null = null;
const idle: (() => void)[] = [];

/** Paths as in the manifest: without the base URL or query. */
function key(url: string): string {
  const path = new URL(url, location.href).pathname;
  const base = import.meta.env.BASE_URL;
  return path.startsWith(base) ? path.slice(base.length) : path;
}

// Every three loader goes through the default manager: note what starts and ends.
// (Its own callbacks only count items, and fire onStart once per busy spell.)
{
  const m = DefaultLoadingManager;
  const start = m.itemStart.bind(m);
  const end = m.itemEnd.bind(m);
  const error = m.itemError.bind(m);
  m.itemStart = (url) => {
    started.add(url);
    start(url);
  };
  m.itemEnd = (url) => {
    finished.add(url);
    end(url);
    if (finished.size >= started.size) for (const f of idle.splice(0)) f();
  };
  m.itemError = (url) => {
    failed ??= key(url);
    error(url);
  };
}

/** Resolves once every file requested so far has loaded. */
export function filesLoaded(): Promise<void> {
  if (finished.size >= started.size) return Promise.resolve();
  return new Promise((resolve) => idle.push(resolve));
}

/** A loader progress callback that feeds the loading screen's byte count. */
export function trackDownload(url: string): (e: ProgressEvent) => void {
  return (e) => received.set(key(url), e.loaded);
}

/** What is being waited for, in the order it's shown when several are. */
export type Stage = "graphics" | "world" | "shaders";

/** Shares of the bar. Downloads and the world are built at the same time. */
const WEIGHT = { graphics: 0.05, files: 0.4, world: 0.35, shaders: 0.2 };

/**
 * The loading screen. Its markup is in index.html, so it shows before any script has
 * loaded; this fills in the progress bar and status line until `finish`, then fades it
 * out over the first frames. Measured: asset downloads (bytes; image sizes come from a
 * build-time manifest, as images report no progress) and world generation (worker jobs
 * outstanding). Shader compilation reports nothing: the bar waits short of the end meanwhile.
 */
export class LoadingScreen {
  private readonly root = document.getElementById("loading")!;
  private readonly bar = this.root.querySelector<HTMLElement>(".bar")!;
  private readonly fill = this.root.querySelector<HTMLElement>(".bar > div")!;
  private readonly status = this.root.querySelector<HTMLElement>(".status")!;
  private stage: Stage = "graphics";
  private world: (() => number) | null = null;
  private worldPeak = 0;
  private worldDone = false;
  /** Progress reached so far, and what the bar shows while it eases there. */
  private reached = 0;
  private shown = 0;
  private lastTick = performance.now();
  private raf = 0;
  private error: string | null = null;

  constructor() {
    const fail = (reason: unknown) =>
      this.fail(reason instanceof Error ? reason.message : String(reason));
    window.addEventListener("error", (e) => fail(e.error ?? e.message));
    window.addEventListener("unhandledrejection", (e) => fail(e.reason));
    this.tick();
  }

  /** Moves on to a stage; `pending` counts the world's outstanding jobs. */
  enter(stage: Stage, pending?: () => number): void {
    if (stage === "shaders") this.worldDone = true;
    this.stage = stage;
    if (pending) this.world = pending;
  }

  /** Everything's in: fades out once the frame being drawn now is on screen. */
  finish(): void {
    if (this.error) return;
    cancelAnimationFrame(this.raf);
    this.show(1, "Ready", true);
    // Two frames: the first is being rendered, the second puts it on screen.
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        this.root.classList.add("done");
        this.root.addEventListener("transitionend", () => this.root.remove(), {
          once: true,
        });
      }),
    );
  }

  get active(): boolean {
    return this.root.isConnected && !this.root.classList.contains("done");
  }

  private fail(message: string): void {
    if (!this.active || this.error) return;
    this.error = message;
    cancelAnimationFrame(this.raf);
    this.root.classList.add("failed");
    this.status.textContent = `Couldn't start: ${message}`;
  }

  private tick = (): void => {
    this.raf = requestAnimationFrame(this.tick);
    if (failed) return this.fail(`couldn't load ${failed}`);

    // Downloads: bytes of the files requested so far that are in the manifest.
    let total = 0;
    let got = 0;
    for (const url of started) {
      const k = key(url);
      const size = __PUBLIC_SIZES__[k];
      if (size === undefined) continue;
      total += size;
      got += finished.has(url) ? size : Math.min(size, received.get(k) ?? 0);
    }
    const files =
      this.stage === "graphics"
        ? 0
        : total
          ? got / total
          : started.size
            ? 1
            : 0;

    // The world: worker jobs done out of the most that were waiting at once.
    let world = this.worldDone ? 1 : 0;
    if (!this.worldDone && this.world) {
      const pending = this.world();
      this.worldPeak = Math.max(this.worldPeak, pending);
      if (this.worldPeak) world = 1 - pending / this.worldPeak;
    }

    let progress =
      (this.stage === "graphics" ? 0 : WEIGHT.graphics) +
      WEIGHT.files * files +
      WEIGHT.world * world;
    let text: string;
    if (this.stage === "graphics") text = "Starting the graphics";
    else if (files < 1) text = `Downloading  ·  ${mb(got)} of ${mb(total)} MB`;
    else if (this.stage === "world") text = "Building the world";
    else {
      text = "Compiling shaders";
      progress = 1 - WEIGHT.shaders;
    }
    this.show(progress, text);
  };

  /**
   * Eases the bar towards `progress`, here rather than with a CSS transition: shader
   * compilation blocks the main thread, and a compositor transition restarted after such
   * a stall can jump back to a stale value. This way the bar only ever pauses.
   */
  private show(progress: number, text: string, instant = false): void {
    // Never backwards (the world's job count can grow while it's being built).
    this.reached = Math.max(this.reached, Math.min(1, progress));
    const now = performance.now();
    const dt = Math.min(0.1, (now - this.lastTick) / 1000);
    this.lastTick = now;
    this.shown = instant
      ? this.reached
      : this.shown + (this.reached - this.shown) * (1 - Math.exp(-12 * dt));
    this.fill.style.transform = `scaleX(${this.shown})`;
    this.bar.setAttribute(
      "aria-valuenow",
      String(Math.round(this.shown * 100)),
    );
    if (this.status.textContent !== text) this.status.textContent = text;
  }
}

function mb(bytes: number): string {
  return (bytes / 1e6).toFixed(1);
}
