import * as THREE from "three/webgpu";
import {
  cos,
  dFdx,
  dFdy,
  dot,
  float,
  floor,
  fract,
  mix,
  pow,
  sin,
  step,
  texture,
  vec2,
  vec3,
  vec4,
} from "three/tsl";

type Float = THREE.Node<"float">;
type Vec2 = THREE.Node<"vec2">;
type Vec3 = THREE.Node<"vec3">;
type Vec4 = THREE.Node<"vec4">;

/** One of the three hex cells blended at a point: where it samples, and its rotation. */
interface Cell {
  uv: Vec2;
  dx: Vec2;
  dy: Vec2;
  cos: Float;
  sin: Float;
}

/**
 * Hex-tiling (Mikkelsen 2022, "Practical Real-Time Hex-Tiling"): texture coordinates `st`
 * split into small hexagons, each showing the texture at a random offset and rotation,
 * blended across their edges. A tiled texture then shows no repeating grid. Each map
 * sampled through it takes three lookups; the blend weights come from one of them (see
 * `weights`) and are shared by the others, so all of a material's maps line up.
 */
export class HexTiling {
  readonly cells: Cell[];
  /** Barycentric weights of the three cells. */
  private readonly bary: Vec3;

  constructor(st: Vec2) {
    const dx = dFdx(st);
    const dy = dFdy(st);
    // A triangle grid (vertices: the hexagons' centres), about a third of a tile across.
    const s = st.mul(2 * Math.sqrt(3));
    const skewed = vec2(s.x.sub(s.y.mul(0.57735027)), s.y.mul(1.15470054));
    const base = floor(skewed);
    const f = fract(skewed);
    const z = float(1).sub(f.x).sub(f.y);
    const upper = step(0, z.negate());
    const sign = upper.mul(2).sub(1);
    this.bary = vec3(
      z.negate().mul(sign),
      upper.sub(f.y.mul(sign)),
      upper.sub(f.x.mul(sign)),
    );
    const vertices = [
      base.add(vec2(upper, upper)),
      base.add(vec2(upper, upper.oneMinus())),
      base.add(vec2(upper.oneMinus(), upper)),
    ];
    this.cells = vertices.map((v) => {
      const h = hash32(v);
      const angle = h.z.mul(2 * Math.PI);
      const c = cos(angle);
      const sn = sin(angle);
      const centre = vec2(v.x.add(v.y.mul(0.5)), v.y.div(1.15470054)).div(
        2 * Math.sqrt(3),
      );
      // Turned about the cell's centre, then shifted at random. (Leaving the centre out
      // keeps the coordinates small, so precise far along the road.)
      const turn = (q: Vec2) =>
        vec2(q.x.mul(c).add(q.y.mul(sn)), q.y.mul(c).sub(q.x.mul(sn)));
      return {
        uv: turn(st.sub(centre)).add(h.xy),
        dx: turn(dx),
        dy: turn(dy),
        cos: c,
        sin: sn,
      };
    });
  }

  /** The three lookups of `map` (gradients from the continuous coordinates: no seams). */
  sample(map: THREE.Texture): Vec4[] {
    return this.cells.map((c) => texture(map, c.uv).grad(c.dx, c.dy));
  }

  /**
   * Blend weights from three lookups' brightness: the cell nearest wins sharply, brighter
   * features over darker ones, so the texture keeps its contrast through the blend.
   */
  weights(samples: Vec4[], contrast = 0.6, sharpness = 7): Vec3 {
    const luma = vec3(0.299, 0.587, 0.114);
    const [a, b, c] = samples as [Vec4, Vec4, Vec4];
    const bright = mix(
      vec3(1),
      vec3(dot(a.rgb, luma), dot(b.rgb, luma), dot(c.rgb, luma)),
      contrast,
    );
    const w = bright.mul(pow(this.bary, vec3(sharpness)));
    return w.div(w.x.add(w.y).add(w.z));
  }

  /** Blend weights from the cells' positions alone (for a lone map). */
  plainWeights(sharpness = 7): Vec3 {
    const w = pow(this.bary, vec3(sharpness));
    return w.div(w.x.add(w.y).add(w.z));
  }

  /** Lookups blended with weights `w`. */
  blend(samples: Vec4[], w: Vec3): Vec4 {
    const [a, b, c] = samples as [Vec4, Vec4, Vec4];
    return a.mul(w.x).add(b.mul(w.y)).add(c.mul(w.z));
  }

  /**
   * A tangent-space normal map (OpenGL, xy scaled by `strength`), each lookup's normal
   * turned back with its cell, blended with weights `w`; unit length.
   */
  normal(map: THREE.Texture, w: Vec3, strength = 1): Vec3 {
    const n = this.sample(map).map((s, i) => {
      const c = this.cells[i]!;
      const t = s.xyz.mul(2).sub(1);
      const x = t.x.mul(strength);
      const y = t.y.mul(strength);
      return vec4(
        x.mul(c.cos).sub(y.mul(c.sin)),
        x.mul(c.sin).add(y.mul(c.cos)),
        t.z,
        0,
      );
    });
    return this.blend(n, w).xyz.normalize();
  }
}

/** Three values in [0, 1) from a 2D point (Dave Hoskins' hash, no sine). */
function hash32(p: Vec2): Vec3 {
  const a = fract(vec3(p.x, p.y, p.x).mul(vec3(0.1031, 0.103, 0.0973)));
  const b = a.add(dot(a, a.yxz.add(33.33)));
  return fract(b.xxy.add(b.yzz).mul(b.zyx));
}
