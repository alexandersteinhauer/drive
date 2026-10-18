/** Small seeded PRNG (mulberry32). Returns floats in [0, 1). */
export function rng(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Deterministic hash of two integers and a seed to [0, 1). */
export function hash2(x: number, y: number, seed: number): number {
  let h =
    Math.imul(x | 0, 0x27d4eb2d) ^
    Math.imul(y | 0, 0x165667b1) ^
    Math.imul(seed | 0, 0x9e3779b9);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

const F2 = 0.5 * (Math.sqrt(3) - 1);
const G2 = (3 - Math.sqrt(3)) / 6;
const GX = [1, -1, 1, -1, 1, -1, 0, 0];
const GY = [1, 1, -1, -1, 0, 0, 1, -1];

/** 2D simplex noise, roughly in [-1, 1]. */
export class Simplex {
  private readonly perm = new Uint8Array(512);

  constructor(seed: number) {
    const random = rng(seed);
    const p = Array.from({ length: 256 }, (_, i) => i);
    for (let i = 255; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [p[i], p[j]] = [p[j]!, p[i]!];
    }
    for (let i = 0; i < 512; i++) this.perm[i] = p[i & 255]!;
  }

  noise(x: number, y: number): number {
    const perm = this.perm;
    const s = (x + y) * F2;
    const i = Math.floor(x + s);
    const j = Math.floor(y + s);
    const t = (i + j) * G2;
    const x0 = x - (i - t);
    const y0 = y - (j - t);
    const i1 = x0 > y0 ? 1 : 0;
    const j1 = 1 - i1;
    const x1 = x0 - i1 + G2;
    const y1 = y0 - j1 + G2;
    const x2 = x0 - 1 + 2 * G2;
    const y2 = y0 - 1 + 2 * G2;
    const ii = i & 255;
    const jj = j & 255;
    let n = 0;
    let c = 0.5 - x0 * x0 - y0 * y0;
    if (c > 0) {
      const g = perm[ii + perm[jj]!]! & 7;
      c *= c;
      n += c * c * (GX[g]! * x0 + GY[g]! * y0);
    }
    c = 0.5 - x1 * x1 - y1 * y1;
    if (c > 0) {
      const g = perm[ii + i1 + perm[jj + j1]!]! & 7;
      c *= c;
      n += c * c * (GX[g]! * x1 + GY[g]! * y1);
    }
    c = 0.5 - x2 * x2 - y2 * y2;
    if (c > 0) {
      const g = perm[ii + 1 + perm[jj + 1]!]! & 7;
      c *= c;
      n += c * c * (GX[g]! * x2 + GY[g]! * y2);
    }
    return 70 * n;
  }

  /** Fractal sum of octaves, normalized to roughly [-1, 1]. */
  fbm(x: number, y: number, octaves: number): number {
    let sum = 0;
    let amp = 1;
    let norm = 0;
    for (let o = 0; o < octaves; o++) {
      sum += amp * this.noise(x, y);
      norm += amp;
      amp *= 0.5;
      x *= 2.03;
      y *= 2.03;
    }
    return sum / norm;
  }
}
