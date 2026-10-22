import * as THREE from "three/webgpu";
import {
  abs,
  attribute,
  Fn,
  If,
  cameraPosition,
  float,
  mix,
  normalWorld,
  positionWorld,
  pow,
  smoothstep,
  texture,
  transformNormalToView,
  vec2,
  vec3,
} from "three/tsl";
import { HexTiling } from "./hexTile";

type Vec2 = THREE.Node<"vec2">;

/**
 * A photo-scanned material (ambientCG, CC0): colour and OpenGL normal map. Roughness is a
 * per-layer constant: its maps add little on terrain, and dropping them keeps the terrain
 * shader within WebGPU's 16-textures-per-stage limit (shadow cascades, sky and haze use the rest).
 */
interface Layer {
  color: THREE.Texture;
  normal: THREE.Texture;
}

const loader = new THREE.TextureLoader();

export function loadLayer(name: string): Layer {
  const load = (suffix: string, srgb: boolean) => {
    const t = loader.load(
      `${import.meta.env.BASE_URL}textures/${name}_${suffix}.webp`,
    );
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = 8;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  };
  return { color: load("color", true), normal: load("normal", false) };
}

let grassLayer: Layer | null = null;
/** The meadow grass layer, shared with the grass blades. */
function meadowLayer(): Layer {
  return (grassLayer ??= loadLayer("grass004"));
}

/**
 * Meadow colour at world position `xz`: the grass texture at two scales, tinted by the
 * "variation" (dry patches) and "forest" (shaded floor) values. `level` picks a fixed mip
 * level, for lookups that are constant over a primitive (grass blades take their root's colour).
 */
export function meadowColor(
  xz: Vec2,
  variation: THREE.Node<"float">,
  forest: THREE.Node<"float">,
  level?: number,
): THREE.Node<"vec3"> {
  const grass = meadowLayer();
  const uv1 = xz.div(3.2);
  // A larger, rotated sample breaks up repetition.
  const uv2 = vec2(
    xz.x.mul(0.8).sub(xz.y.mul(0.6)),
    xz.x.mul(0.6).add(xz.y.mul(0.8)),
  ).div(11.3);
  const at = (uv: Vec2) =>
    level === undefined
      ? texture(grass.color, uv).rgb
      : texture(grass.color, uv).level(float(level)).rgb;
  const dryTint = mix(vec3(1), vec3(1.25, 1.12, 0.72), variation.mul(0.6)).mul(
    mix(vec3(1), vec3(0.55, 0.6, 0.45), forest.mul(0.85)),
  );
  return mix(at(uv1), at(uv2), 0.4).mul(dryTint);
}

/** Samples a tangent-space normal map, decoded to [-1, 1], with xy scaled by `strength`. */
function decode(t: THREE.Texture, uv: Vec2, strength: number) {
  const s = texture(t, uv).rgb.mul(2).sub(1);
  return vec3(s.x.mul(strength), s.y.mul(strength), s.z);
}

/**
 * Terrain shading, all in world space so neighbouring tiles match:
 *  - meadow grass, sampled at two scales to hide tiling, tinted by the per-vertex "variation"
 *    attribute (large dry/green patches)
 *  - dirt on moderate slopes and on the verge next to the road (per-vertex "road" distance)
 *  - rock on steep cuttings, projected from three axes so it doesn't smear on steep faces
 * Normal maps are blended per layer and applied with UDN (whiteout-style) reorientation.
 */
export function terrainMaterial(
  roadEdge: number,
): THREE.MeshStandardNodeMaterial {
  const grass = meadowLayer();
  const dirt = loadLayer("ground103");
  const rock = loadLayer("rock030");
  const gravel = loadLayer("gravel040");

  const p = positionWorld;
  const n = normalWorld.normalize();
  const uvTop = vec2(p.x, p.z);
  const grassUv = uvTop.div(3.2);
  const dirtUv = uvTop.div(3.5);
  const gravelUv = uvTop.div(2.5);
  // Dirt and gravel (verges, cuttings, forest floor) hex-tiled, so their tiles don't
  // repeat visibly along the road.
  const dirtHex = new HexTiling(dirtUv);
  const gravelHex = new HexTiling(gravelUv);

  const variation = attribute("variation", "float");
  // Forest density: shaded, leaf-littered floor under the woods, which also marks them out
  // on distant hills beyond the trees' range.
  const forest = attribute("forest", "float");
  const road = attribute("road", "float");
  const slope = float(1).sub(n.y);

  // Layer weights. Slope is 1 - n.y: 0.13 is ~30 degrees, 0.29 is ~45 degrees.
  // A low-frequency texture lookup gives a noise value to break up straight edges.
  // Rock is the highest-contrast texture; sampled very large it makes a good ragged noise.
  const noise = texture(rock.color, uvTop.div(23)).r.mul(4).sub(1.2);
  const verge = float(1).sub(
    smoothstep(
      roadEdge + 0.2,
      float(roadEdge + 2).add(noise.mul(2).max(0)),
      road,
    ),
  );
  const dirtW = smoothstep(0.07, 0.17, slope.add(noise.mul(0.03)))
    .max(verge.mul(0.55))
    .max(forest.mul(0.35))
    .max(smoothstep(0.82, 1, variation).mul(0.5));
  const rockW = smoothstep(0.16, 0.27, slope.add(noise.mul(0.04)));
  // Loose gravel spilling off the shoulder onto the verge, with a ragged edge.
  const gravelW = float(1).sub(
    smoothstep(
      roadEdge + 0.3,
      float(roadEdge + 1.2).add(noise.mul(1.4).max(0)),
      road,
    ),
  );

  // Triplanar blend weights for rock.
  const tw = pow(abs(n), vec3(8));
  const tri = tw.div(tw.x.add(tw.y).add(tw.z));
  const rockScale = 5;
  const rx = vec2(p.z, p.y).div(rockScale);
  const ry = vec2(p.x, p.z).div(rockScale);
  const rz = vec2(p.x, p.y).div(rockScale);

  // Layers are only sampled where they show (most of the ground is plain meadow): the
  // weights pick branches, so a pixel pays for the textures it uses. Their edges are soft
  // blends, so the threshold doesn't show.
  const SHOWS = 0.002;
  const hasDirt = dirtW.greaterThan(SHOWS);
  const hasRock = rockW.greaterThan(SHOWS);
  const hasGravel = gravelW.greaterThan(SHOWS);

  // Colour.
  const color = Fn(() => {
    const c = meadowColor(uvTop, variation, forest).toVar();
    If(hasDirt, () => {
      const d = dirtHex.sample(dirt.color);
      c.assign(mix(c, dirtHex.blend(d, dirtHex.weights(d)).rgb, dirtW));
    });
    If(hasRock, () => {
      const rockColor = texture(rock.color, rx)
        .rgb.mul(tri.x)
        .add(texture(rock.color, ry).rgb.mul(tri.y))
        .add(texture(rock.color, rz).rgb.mul(tri.z));
      // A touch more contrast on the rock so its structure reads from the road.
      // Large-scale variation over the rock face hides the scan's repeating streaks.
      const rockMacro = texture(rock.color, ry.div(4.3)).r.mul(1.6).add(0.5);
      // Lighter, greyer and a little more contrasty than the scan, so cuttings read as rock.
      const rockLum = rockColor.dot(vec3(0.2126, 0.7152, 0.0722));
      const rockGraded = mix(
        rockColor.mul(rockMacro),
        vec3(rockLum.mul(rockMacro)),
        0.35,
      )
        .sub(0.08)
        .mul(1.9)
        .add(0.12)
        .max(0);
      c.assign(mix(c, rockGraded, rockW));
    });
    return c;
  })();

  // Roughness: per-layer constants.
  const roughness = mix(
    mix(mix(float(0.8), 0.85, dirtW), 0.85, gravelW),
    0.75,
    rockW,
  );

  // Normals. Fine detail fades out with distance, where it only adds shimmer.
  const detail = float(1).sub(smoothstep(25, 140, p.distance(cameraPosition)));
  const world = Fn(() => {
    const w = n.toVar();
    If(detail.greaterThan(0), () => {
      const t = decode(grass.normal, grassUv, 0.9).toVar();
      If(hasDirt, () => {
        t.assign(
          mix(t, dirtHex.normal(dirt.normal, dirtHex.plainWeights()), dirtW),
        );
      });
      If(hasGravel, () => {
        t.assign(
          mix(
            t,
            gravelHex.normal(gravel.normal, gravelHex.plainWeights()),
            gravelW,
          ),
        );
      });
      t.mulAssign(vec3(detail, detail, 1));
      // UDN on the top projection: tangent x -> world x, tangent y -> world z.
      w.assign(vec3(t.x.add(n.x), n.y, t.y.add(n.z)).normalize());
    });
    If(hasRock, () => {
      const tx = decode(rock.normal, rx, 1);
      const ty = decode(rock.normal, ry, 1);
      const tz = decode(rock.normal, rz, 1);
      const rockWorld = vec3(n.x, tx.y.add(n.y), tx.x.add(n.z))
        .mul(tri.x)
        .add(vec3(ty.x.add(n.x), n.y, ty.y.add(n.z)).mul(tri.y))
        .add(vec3(tz.x.add(n.x), tz.y.add(n.y), n.z).mul(tri.z))
        .normalize();
      w.assign(mix(w, rockWorld, rockW).normalize());
    });
    return w;
  })();

  const material = new THREE.MeshStandardNodeMaterial();
  material.colorNode = color;
  material.roughnessNode = roughness;
  material.normalNode = transformNormalToView(world);
  material.metalness = 0;
  return material;
}
