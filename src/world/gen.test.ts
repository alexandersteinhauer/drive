import { describe, expect, test } from "bun:test";
import { grassChunk } from "./gen";
import { Landscape } from "./landscape";

describe("grass chunks", () => {
  const land = new Landscape(1);
  land.road.ensure(2000);

  test("a sparser chunk is the start of the full one", () => {
    // A chunk beside the road near its start, with a meadow in it.
    const spots = [-1, 0].flatMap((cx) =>
      [-2, -3, -4, -5].map((cz) => [cx, cz] as const),
    );
    const [cx, cz] = spots.find(
      ([x, z]) => (grassChunk(land, x, z)?.count ?? 0) > 500,
    )!;
    const full = grassChunk(land, cx, cz)!;
    for (const maxRank of [1 / 8, 1 / 4, 1 / 2]) {
      const part = grassChunk(land, cx, cz, maxRank)!;
      const n = part.count;
      expect(n).toBe(
        [...Array(full.count).keys()].filter(
          (i) => full.info[i * 4 + 1]! <= maxRank,
        ).length,
      );
      expect(part.root).toEqual(full.root.subarray(0, n * 4));
      expect(part.info).toEqual(full.info.subarray(0, n * 4));
      expect(part.ground).toEqual(full.ground.subarray(0, n * 2));
    }
  });
});
