import type { GrassResult, TerrainRequest, TerrainResult } from "./world/gen";
import type { Job, Request } from "./world/worker";
import type { Landscape } from "./world/landscape";

type Result<J extends Job> = J["kind"] extends "terrain"
  ? TerrainResult
  : J["kind"] extends "grass"
    ? GrassResult | null
    : { heights: Float32Array; tops: Float32Array };

/**
 * A pool of world workers (world/worker.ts): terrain tiles, grass chunks and sun-field
 * tiles are generated off the main thread, which only wraps the arrays in meshes. Each
 * worker runs its jobs in order; jobs go to the least busy one.
 */
export class WorldWorkers {
  private readonly workers: { worker: Worker; busy: number }[] = [];
  private readonly waiting = new Map<number, (result: unknown) => void>();
  private nextId = 1;

  constructor(private readonly land: Landscape) {
    const n = Math.min(
      4,
      Math.max(1, (navigator.hardwareConcurrency || 4) - 2),
    );
    for (let i = 0; i < n; i++) {
      const worker = new Worker(new URL("./world/worker.ts", import.meta.url), {
        type: "module",
        // The worker builds its own copy of the landscape from this seed.
        name: String(land.seed),
      });
      const slot = { worker, busy: 0 };
      worker.addEventListener(
        "message",
        (e: MessageEvent<{ id: number; result: unknown }>) => {
          slot.busy--;
          const done = this.waiting.get(e.data.id)!;
          this.waiting.delete(e.data.id);
          done(e.data.result);
        },
      );
      this.workers.push(slot);
    }
  }

  /** Jobs sent and not yet answered. */
  get pending(): number {
    return this.waiting.size;
  }

  /** Workers in the pool. */
  get size(): number {
    return this.workers.length;
  }

  run(job: { kind: "terrain" } & TerrainRequest): Promise<TerrainResult>;
  run(job: {
    kind: "grass";
    cx: number;
    cz: number;
    maxRank: number;
  }): Promise<GrassResult | null>;
  run(job: {
    kind: "sun";
    tiles: number[];
    size: number;
    texel: number;
  }): Promise<{ heights: Float32Array; tops: Float32Array }>;
  run<J extends Job>(job: J): Promise<Result<J>> {
    const id = this.nextId++;
    const slot = this.workers.reduce((a, b) => (b.busy < a.busy ? b : a));
    slot.busy++;
    const request: Request = {
      id,
      progress: this.land.road.finalProgress,
      job,
    };
    return new Promise((resolve) => {
      this.waiting.set(id, resolve as (r: unknown) => void);
      slot.worker.postMessage(request);
    });
  }
}
