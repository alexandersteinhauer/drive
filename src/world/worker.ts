/// <reference lib="webworker" />
import { grassChunk, sunTiles, terrainTile, type TerrainRequest } from "./gen";
import { Landscape } from "./landscape";

/**
 * A world worker: its own copy of the landscape (the road is deterministic, sample by
 * sample, so it matches the main thread's exactly), generating on request.
 */
export type Job =
  | ({ kind: "terrain" } & TerrainRequest)
  | { kind: "grass"; cx: number; cz: number; maxRank: number }
  | { kind: "sun"; tiles: number[]; size: number; texel: number };

export interface Request {
  id: number;
  /** The main thread's final road progress: generated up to there first. */
  progress: number;
  job: Job;
}

const land = new Landscape(Number(self.name));

self.addEventListener("message", (e: MessageEvent<Request>) => {
  const { id, progress, job } = e.data;
  land.road.ensure(progress);
  let result: unknown;
  const transfer: Transferable[] = [];
  const keep = (...arrays: { buffer: ArrayBufferLike }[]) => {
    for (const a of arrays) transfer.push(a.buffer as ArrayBuffer);
  };
  if (job.kind === "terrain") {
    const r = terrainTile(land, job);
    keep(r.position, r.normal, r.variation, r.road, r.forest, r.index);
    for (const [, items] of r.trees) keep(items);
    result = r;
  } else if (job.kind === "grass") {
    const r = grassChunk(land, job.cx, job.cz, job.maxRank);
    if (r) keep(r.root, r.info, r.ground);
    result = r;
  } else {
    const r = sunTiles(land, job.tiles, job.size, job.texel);
    keep(r.heights, r.tops);
    result = r;
  }
  (self as unknown as Worker).postMessage({ id, result }, transfer);
});
