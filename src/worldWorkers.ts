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
  private readonly pending = new Map<number, (result: unknown) => void>();
  private nextId = 1;

  constructor(private readonly land: Landscape) {
    const n = Math.min(
      4,
      Math.max(1, (navigator.hardwareConcurrency || 4) - 2),
    );
    for (let i = 0; i < n; i++) {
      const worker = new Worker(new URL("./world/worker.ts", import.meta.url), {
        type: "module",
      });
      const slot = { worker, busy: 0 };
      worker.addEventListener(
        "message",
        (e: MessageEvent<{ id: number; result: unknown }>) => {
          slot.busy--;
          const done = this.pending.get(e.data.id)!;
          this.pending.delete(e.data.id);
          done(e.data.result);
        },
      );
      this.workers.push(slot);
    }
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
      this.pending.set(id, resolve as (r: unknown) => void);
      slot.worker.postMessage(request);
    });
  }
}
