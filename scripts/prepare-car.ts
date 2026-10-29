/**
 * Turns a downloaded car GLB into a game-ready asset:
 *   bun scripts/prepare-car.ts bmw-i4 <source.glb> public/models/bmw-i4.glb
 *   bun scripts/prepare-car.ts vw-beetle <source.glb> public/models/vw-beetle.glb
 *
 * - drops parts the game doesn't show (the BMW's interior and licence plate)
 * - bakes all transforms, scales to metres (by overall length) and turns the car to face -Z
 * - splits wheel geometry per corner into a spinning part (tyre, rim, disc, nuts) and a
 *   steering-only part (caliper), each centred on its wheel centre
 * - merges the body by material, simplifies, resizes textures and meshopt-compresses
 *
 * Output: nodes "body", "wheel_<fl|fr|rl|rr>_spin" and "wheel_<..>_caliper"; origin on the
 * ground, midway between the axles; extras.wheels holds the measured wheel centres/radius.
 */
import {
  NodeIO,
  Primitive,
  type Document,
  type Transform,
} from "@gltf-transform/core";
import {
  EXTMeshoptCompression,
  KHRMeshQuantization,
  ALL_EXTENSIONS,
} from "@gltf-transform/extensions";
import {
  dedup,
  dequantize,
  flatten,
  join,
  prune,
  resample,
  compactPrimitive,
  textureCompress,
  transformPrimitive,
  weld,
  quantize,
} from "@gltf-transform/functions";
import { MeshoptEncoder, MeshoptSimplifier } from "meshoptimizer";
import sharp from "sharp";

/** What differs between the source models. Parts are matched by node and material name. */
interface Profile {
  /** Overall length (m) the model is scaled to. */
  length: number;
  /** Parts removed. */
  drop: (node: string, material: string) => boolean;
  /** Parts whose bounds give the wheel centres and radius. */
  tyre: (node: string, material: string) => boolean;
  /** Parts that only steer with their wheel rather than spin (brake calipers). */
  steerOnly: (node: string, material: string) => boolean;
  /**
   * Parts that make up the wheels, if the source has them separate; otherwise wheel parts
   * are found by position, inside each tyre's cylinder.
   */
  wheel?: (node: string, material: string) => boolean;
  /** Parts every triangle of which must end up in a wheel. */
  wheelOnly: (node: string, material: string) => boolean;
  /** Simplify the meshes (the BMW's are far denser than the game needs). */
  simplify: boolean;
  /**
   * Fill each texture outside its UV islands from the islands' own edges. Atlases with a
   * plain background otherwise bleed it into the parts' edges as the texture is shrunk and
   * mipmapped: the Beetle's tyres got a bright, metallic rim from it.
   */
  padTextures: boolean;
  /** Last fix-ups to the materials, once the parts are split. */
  materials?: (doc: Document) => void;
}

const PROFILES: Record<string, Profile> = {
  "bmw-i4": {
    length: 4.783,
    drop: (_, m) => /Interior|License_Plate/.test(m),
    tyre: (_, m) => /Tires/.test(m),
    steerOnly: (_, m) => /Calipers/.test(m),
    wheelOnly: (_, m) => /Tires|Rims|BrakeRotors/.test(m),
    simplify: true,
    padTextures: false,
  },
  // 1963 Beetle, 4070 mm over the bumpers. A light model with an interior, which shows
  // through its windows; its tyres, glass and underbody share one alpha-blended material.
  "vw-beetle": {
    length: 4.07,
    drop: () => false,
    tyre: (n) => /^tires?_/.test(n),
    // The tyre nodes include the hubcaps; the cylinder would also catch fender lips.
    wheel: (n) => /^tires?_/.test(n),
    steerOnly: () => false,
    wheelOnly: (n) => /^tires?_/.test(n),
    simplify: false,
    padTextures: true,
    materials: (doc) => {
      const names: Record<string, string> = {
        UDIM1001: "Bumpers",
        UDIM1002: "Trim",
        UDIM1003: "Tires",
        UDIM1004: "InteriorParts",
        UDIM1005: "Body",
        UDIM1006: "Interior",
        UDIM1007: "BodyPanels",
      };
      const root = doc.getRoot();
      for (const m of root.listMaterials())
        m.setName(names[m.getName()] ?? m.getName());
      const shared = root.listMaterials().find((m) => m.getName() === "Tires")!;
      // Only the windows need blending; drawn opaque, the rest sorts and shadows properly.
      const glass = shared.clone().setName("WindowGlass");
      shared.setAlphaMode("OPAQUE");
      for (const mesh of root.listMeshes())
        for (const prim of mesh.listPrimitives())
          if (prim.getMaterial() === shared && windows.has(prim))
            prim.setMaterial(glass);
    },
  },
};

const [profileName, srcPath, outPath] = process.argv.slice(2);
const profile = PROFILES[profileName ?? ""];
if (!profile || !srcPath || !outPath)
  throw new Error(
    `usage: prepare-car.ts <${Object.keys(PROFILES).join("|")}> <source.glb> <out.glb>`,
  );
const LENGTH = profile.length;
/** Primitives of the Beetle's window node, tracked through the split. */
const windows = new Set<Primitive>();

await MeshoptEncoder.ready;
await MeshoptSimplifier.ready;
const io = new NodeIO()
  .registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ "meshopt.encoder": MeshoptEncoder });
const doc = await io.read(srcPath);
const root = doc.getRoot();
const scene = root.listScenes()[0]!;

// Bake every node's world transform into its geometry, so all positions share one frame.
await doc.transform(flatten());
for (const node of root.listNodes()) {
  const mesh = node.getMesh();
  if (!mesh) continue;
  const m = node.getWorldMatrix();
  for (const prim of mesh.listPrimitives()) transformPrimitive(prim, m);
}
for (const node of root.listNodes())
  node.setMatrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

const nodeOf = new Map(
  root
    .listNodes()
    .filter((n) => n.getMesh())
    .map((n) => [n.getMesh()!, n.getName()]),
);
const allPrims = () =>
  root.listMeshes().flatMap((mesh) =>
    mesh.listPrimitives().map((prim) => ({
      mesh,
      prim,
      node: nodeOf.get(mesh) ?? "",
      material: prim.getMaterial()?.getName() ?? "",
    })),
  );
for (const { mesh, prim, node, material } of allPrims())
  if (profile.drop(node, material)) {
    mesh.removePrimitive(prim);
    prim.dispose();
  }

// Most materials are untextured; their (per-triangle, slightly different) UVs only stop
// vertices from being welded, which simplification needs.
for (const { prim } of allPrims()) {
  const m = prim.getMaterial();
  const textured =
    m &&
    (m.getBaseColorTexture() ||
      m.getNormalTexture() ||
      m.getMetallicRoughnessTexture() ||
      m.getEmissiveTexture() ||
      m.getOcclusionTexture());
  if (!textured)
    for (const sem of prim.listSemantics())
      if (sem.startsWith("TEXCOORD") || sem === "TANGENT")
        prim.setAttribute(sem, null);
}

// Overall bounds after dropping, to find scale and centre.
const lo = [Infinity, Infinity, Infinity];
const hi = [-Infinity, -Infinity, -Infinity];
const eachPos = (
  prim: Primitive,
  fn: (x: number, y: number, z: number) => void,
) => {
  const pos = prim.getAttribute("POSITION")!;
  const v = [0, 0, 0];
  for (let i = 0; i < pos.getCount(); i++) {
    pos.getElement(i, v);
    fn(v[0]!, v[1]!, v[2]!);
  }
};
for (const { prim } of allPrims())
  eachPos(prim, (x, y, z) => {
    [x, y, z].forEach((c, i) => {
      lo[i] = Math.min(lo[i]!, c);
      hi[i] = Math.max(hi[i]!, c);
    });
  });
const scale = LENGTH / (hi[2]! - lo[2]!);
const cx = (lo[0]! + hi[0]!) / 2;
const cz = (lo[2]! + hi[2]!) / 2;
// The source faces +Z (headlights at its max Z); rotate 180 deg about Y to face -Z, scale to metres.
const toMetres = [
  -scale,
  0,
  0,
  0,
  0,
  scale,
  0,
  0,
  0,
  0,
  -scale,
  0,
  cx * scale,
  -lo[1]! * scale,
  cz * scale,
  1,
];
for (const { prim } of allPrims()) transformPrimitive(prim, toMetres);

// Wheel centres and radius from the tyre geometry, per corner.
const tyres = allPrims().filter(({ node, material }) =>
  profile.tyre(node, material),
);
const corners = ["fl", "fr", "rl", "rr"] as const;
type Corner = (typeof corners)[number];
const cornerOf = (x: number, z: number): Corner =>
  z < 0 ? (x < 0 ? "fl" : "fr") : x < 0 ? "rl" : "rr";
const tb = Object.fromEntries(
  corners.map((c) => [
    c,
    {
      lo: [Infinity, Infinity, Infinity],
      hi: [-Infinity, -Infinity, -Infinity],
    },
  ]),
);
for (const { prim } of tyres)
  eachPos(prim, (x, y, z) => {
    const b = tb[cornerOf(x, z)]!;
    [x, y, z].forEach((c, i) => {
      b.lo[i] = Math.min(b.lo[i]!, c);
      b.hi[i] = Math.max(b.hi[i]!, c);
    });
  });
const wheels = Object.fromEntries(
  corners.map((c) => {
    const b = tb[c]!;
    return [
      c,
      {
        center: [0, 1, 2].map((i) => (b.lo[i]! + b.hi[i]!) / 2),
        radius: (b.hi[1]! - b.lo[1]!) / 2,
        inner: c.endsWith("l") ? b.hi[0]! : b.lo[0]!,
        outer: c.endsWith("l") ? b.lo[0]! : b.hi[0]!,
      },
    ];
  }),
) as Record<
  Corner,
  { center: number[]; radius: number; inner: number; outer: number }
>;
const radius = corners.reduce((s, c) => s + wheels[c].radius, 0) / 4;
const frontZ = (wheels.fl.center[2]! + wheels.fr.center[2]!) / 2;
const rearZ = (wheels.rl.center[2]! + wheels.rr.center[2]!) / 2;
// Re-centre on the axle midpoint so the asset lines up with the physics by axle positions.
const mid = (frontZ + rearZ) / 2;
for (const { prim } of allPrims())
  transformPrimitive(prim, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -mid, 1]);
for (const c of corners) wheels[c].center[2]! -= mid;
console.log(
  `scale ${scale.toFixed(5)}, wheelbase ${(rearZ - frontZ).toFixed(3)} m, track ${(wheels.fr.center[0]! - wheels.fl.center[0]!).toFixed(3)}/${(wheels.rr.center[0]! - wheels.rl.center[0]!).toFixed(3)} m, tyre radius ${radius.toFixed(3)} m`,
);

/** Which wheel a triangle belongs to (by its centroid inside the wheel's cylinder), or null. */
function wheelOf(x: number, y: number, z: number): Corner | null {
  const c = cornerOf(x, z);
  const w = wheels[c];
  const r = Math.hypot(y - w.center[1]!, z - w.center[2]!);
  const lat = c.endsWith("l") ? w.inner - x : x - w.inner; // > 0 outboard of the tyre's inner face
  const width = Math.abs(w.outer - w.inner);
  // Up to just past the tread; nothing else in the lateral band reaches that far in.
  return r < 1.04 * w.radius && lat > -0.03 && lat < width + 0.04 ? c : null;
}

/** Copies the given triangles of `prim` into a new non-indexed primitive with the same material. */
function extract(prim: Primitive, tris: number[]): Primitive {
  const idx = prim.getIndices();
  const vertex = (t: number, k: number) =>
    idx ? idx.getScalar(t * 3 + k) : t * 3 + k;
  const out = doc.createPrimitive().setMaterial(prim.getMaterial());
  for (const semantic of prim.listSemantics()) {
    const src = prim.getAttribute(semantic)!;
    const size = src.getElementSize();
    const data = new Float32Array(tris.length * 3 * size);
    const el: number[] = [];
    tris.forEach((t, i) => {
      for (let k = 0; k < 3; k++) {
        src.getElement(vertex(t, k), el);
        data.set(el, (i * 3 + k) * size);
      }
    });
    out.setAttribute(
      semantic,
      doc
        .createAccessor()
        .setType(src.getType())
        .setArray(data)
        .setBuffer(src.getBuffer()),
    );
  }
  return out;
}

// Split every primitive's triangles into body / per-wheel spin / per-wheel caliper groups.
const groups = new Map<string, Primitive[]>();
const add = (key: string, prim: Primitive) =>
  groups.set(key, [...(groups.get(key) ?? []), prim]);
for (const { prim, node, material } of allPrims()) {
  const pos = prim.getAttribute("POSITION")!;
  const idx = prim.getIndices();
  const count = (idx ? idx.getCount() : pos.getCount()) / 3;
  const buckets = new Map<string, number[]>();
  const a = [0, 0, 0];
  const b = [0, 0, 0];
  const c = [0, 0, 0];
  const isStatic = profile.steerOnly(node, material);
  for (let t = 0; t < count; t++) {
    pos.getElement(idx ? idx.getScalar(t * 3) : t * 3, a);
    pos.getElement(idx ? idx.getScalar(t * 3 + 1) : t * 3 + 1, b);
    pos.getElement(idx ? idx.getScalar(t * 3 + 2) : t * 3 + 2, c);
    const x = (a[0]! + b[0]! + c[0]!) / 3;
    const y = (a[1]! + b[1]! + c[1]!) / 3;
    const z = (a[2]! + b[2]! + c[2]!) / 3;
    const w = isStatic
      ? cornerOf(x, z)
      : profile.wheel
        ? profile.wheel(node, material)
          ? cornerOf(x, z)
          : null
        : wheelOf(x, y, z);
    const key = w ? `wheel_${w}_${isStatic ? "caliper" : "spin"}` : "body";
    buckets.set(key, [...(buckets.get(key) ?? []), t]);
  }
  for (const [key, tris] of buckets) {
    // Every triangle of these parts belongs to a wheel; any left in the body means the
    // wheel classification above is wrong (it would neither spin nor steer).
    if (key === "body" && profile.wheelOnly(node, material))
      throw new Error(
        `${tris.length} triangles of ${node} (${material}) left in the body`,
      );
    const part = extract(prim, tris);
    if (node.startsWith("windows")) windows.add(part);
    add(key, part);
  }
}

// Rebuild the scene: one node per group, wheel parts centred on their wheel centre.
for (const node of root.listNodes()) node.dispose();
for (const mesh of root.listMeshes()) mesh.dispose();
for (const [key, prims] of groups) {
  const mesh = doc.createMesh(key);
  const node = doc.createNode(key).setMesh(mesh);
  const corner = key.split("_")[1] as Corner | undefined;
  const center = key === "body" ? [0, 0, 0] : wheels[corner!].center;
  for (const prim of prims) {
    transformPrimitive(prim, [
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      -center[0]!,
      -center[1]!,
      -center[2]!,
      1,
    ]);
    mesh.addPrimitive(prim);
  }
  node.setTranslation(center as [number, number, number]);
  scene.addChild(node);
}
scene.setExtras({ wheels, radius, wheelbase: rearZ - frontZ });
profile.materials?.(doc);

/**
 * Simplification: stop at this error (relative to each mesh's size, counting both shape and
 * normals) or this fraction of the triangles. The error is what limits it; glossy paint shows
 * any bent normal as a dent. Normals are kept at 16 bits for the same reason.
 */
const SIMPLIFY_ERROR = 0.0002;
const TEXTURE_SIZE = 1024;
const SIMPLIFY_RATIO = 0.25;
const NORMAL_WEIGHT = 1;
const NORMAL_BITS = 16;
if (profile.padTextures) await padTextures(doc, TEXTURE_SIZE);
const before = triangles();
await doc.transform(
  dedup(),
  join({ keepNamed: true }),
  // The source stores nearly (not bitwise) identical vertices per triangle; snapping them to a
  // quantization grid first lets weld merge them, which simplify needs.
  quantize({
    quantizePosition: 14,
    quantizeNormal: NORMAL_BITS,
    quantizeTexcoord: 12,
  }),
  dequantize(),
  weld(),
  ...(profile.simplify
    ? [shadedSimplify(SIMPLIFY_RATIO, SIMPLIFY_ERROR, NORMAL_WEIGHT)]
    : []),
  textureCompress({
    targetFormat: "webp",
    resize: [TEXTURE_SIZE, TEXTURE_SIZE],
  }),
  prune(),
  resample(),
  quantize({ quantizeNormal: NORMAL_BITS }),
);
doc
  .createExtension(EXTMeshoptCompression)
  .setRequired(true)
  .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
doc.createExtension(KHRMeshQuantization).setRequired(true);
console.log(
  `triangles ${before} -> ${triangles()}, nodes ${[...groups.keys()].join(", ")}`,
);
await io.write(outPath, doc);

/**
 * Simplifies every triangle mesh with meshoptimizer, keeping both the shape and the shading
 * normals within the error. glTF-Transform's simplify() only looks at positions: it kept the
 * shape within millimetres but bent the normals across the larger triangles (median 6 deg on
 * the paint), and glossy paint showed that as dents.
 */
function shadedSimplify(
  ratio: number,
  error: number,
  normalWeight: number,
): Transform {
  return (document) => {
    for (const mesh of document.getRoot().listMeshes())
      for (const prim of mesh.listPrimitives()) {
        const indices = prim.getIndices();
        const position = prim.getAttribute("POSITION");
        const normal = prim.getAttribute("NORMAL");
        if (
          prim.getMode() !== Primitive.Mode.TRIANGLES ||
          !indices ||
          !position ||
          !normal
        )
          continue;
        const src = new Uint32Array(indices.getArray()!);
        const target = Math.floor((ratio * src.length) / 3) * 3;
        const [dst] = MeshoptSimplifier.simplifyWithAttributes(
          src,
          new Float32Array(position.getArray()!),
          3,
          new Float32Array(normal.getArray()!),
          3,
          [normalWeight, normalWeight, normalWeight],
          null,
          target,
          error,
        );
        prim.setIndices(
          document.createAccessor().setType("SCALAR").setArray(dst),
        );
        if (indices.listParents().length === 1) indices.dispose();
        compactPrimitive(prim);
      }
  };
}

function triangles(): number {
  return allPrims().reduce(
    (s, { prim }) =>
      s +
      (prim.getIndices()?.getCount() ??
        prim.getAttribute("POSITION")!.getCount()) /
        3,
    0,
  );
}

/**
 * Shrinks every texture to at most `size` and fills the texels outside its UV islands (by
 * push-pull: averaging the covered texels into ever coarser levels, then filling the gaps
 * from them), so that no background colour is left to blend into the islands' edges.
 */
async function padTextures(document: Document, size: number): Promise<void> {
  for (const texture of document.getRoot().listTextures()) {
    const image = texture.getImage();
    if (!image) continue;
    const { hasAlpha } = await sharp(image).metadata();
    const { data, info } = await sharp(image)
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    const { width, height } = info;
    const covered = coverage(document, texture, width, height);
    // Box-filter down to the output size, averaging covered texels only.
    let w = width;
    let h = height;
    let factor = 1;
    while (w > size || h > size) {
      w /= 2;
      h /= 2;
      factor *= 2;
    }
    const color = new Float32Array(w * h * 4);
    const weight = new Float32Array(w * h);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        if (!covered[i]) continue;
        const o = Math.floor(y / factor) * w + Math.floor(x / factor);
        for (let c = 0; c < 4; c++) color[o * 4 + c]! += data[i * 4 + c]!;
        weight[o]! += 1;
      }
    for (let o = 0; o < w * h; o++)
      if (weight[o]) {
        for (let c = 0; c < 4; c++) color[o * 4 + c]! /= weight[o]!;
        weight[o] = 1;
      }
    pushPull(color, weight, w, h);
    const out = Buffer.alloc(w * h * 4);
    for (let i = 0; i < out.length; i++)
      out[i] = Math.round(Math.min(255, Math.max(0, color[i]!)));
    let img = sharp(out, { raw: { width: w, height: h, channels: 4 } });
    if (!hasAlpha) img = img.removeAlpha();
    texture.setImage(await img.png().toBuffer()).setMimeType("image/png");
  }
}

/** Which texels of a `width` x `height` texture the triangles using it cover (1 texel margin). */
function coverage(
  document: Document,
  texture: ReturnType<Document["createTexture"]>,
  width: number,
  height: number,
): Uint8Array {
  const covered = new Uint8Array(width * height);
  const users = new Set(
    texture.listParents().filter((p) => p.propertyType === "Material"),
  );
  const a = [0, 0];
  const b = [0, 0];
  const c = [0, 0];
  for (const mesh of document.getRoot().listMeshes())
    for (const prim of mesh.listPrimitives()) {
      if (!users.has(prim.getMaterial()!)) continue;
      const uv = prim.getAttribute("TEXCOORD_0");
      if (!uv) continue;
      const idx = prim.getIndices();
      const count = (idx ? idx.getCount() : uv.getCount()) / 3;
      const at = (t: number, k: number, out: number[]) => {
        uv.getElement(idx ? idx.getScalar(t * 3 + k) : t * 3 + k, out);
        out[0] = (out[0]! - Math.floor(out[0]!)) * width;
        out[1] = (out[1]! - Math.floor(out[1]!)) * height;
      };
      for (let t = 0; t < count; t++) {
        at(t, 0, a);
        at(t, 1, b);
        at(t, 2, c);
        rasterize(a, b, c, width, height, covered);
      }
    }
  return covered;
}

/** Marks the texels a UV triangle touches, grown by about a texel so its edges are kept. */
function rasterize(
  a: number[],
  b: number[],
  c: number[],
  width: number,
  height: number,
  out: Uint8Array,
): void {
  const x0 = Math.max(0, Math.floor(Math.min(a[0]!, b[0]!, c[0]!) - 1));
  const x1 = Math.min(width - 1, Math.ceil(Math.max(a[0]!, b[0]!, c[0]!) + 1));
  const y0 = Math.max(0, Math.floor(Math.min(a[1]!, b[1]!, c[1]!) - 1));
  const y1 = Math.min(height - 1, Math.ceil(Math.max(a[1]!, b[1]!, c[1]!) + 1));
  const area =
    (b[0]! - a[0]!) * (c[1]! - a[1]!) - (b[1]! - a[1]!) * (c[0]! - a[0]!);
  if (area === 0) return;
  const sign = Math.sign(area);
  // Signed distance (in texels) of a point to each edge; inside if all are above -margin.
  const edges = [
    [a, b],
    [b, c],
    [c, a],
  ].map(([p, q]) => {
    const dx = q![0]! - p![0]!;
    const dy = q![1]! - p![1]!;
    const len = Math.hypot(dx, dy) || 1;
    return { p: p!, dx: (dx / len) * sign, dy: (dy / len) * sign };
  });
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) {
      const px = x + 0.5;
      const py = y + 0.5;
      if (
        edges.every((e) => e.dx * (py - e.p[1]!) - e.dy * (px - e.p[0]!) > -1.2)
      )
        out[y * width + x] = 1;
    }
}

/** Fills texels with zero weight from the weighted average of ever larger neighbourhoods. */
function pushPull(
  color: Float32Array,
  weight: Float32Array,
  w: number,
  h: number,
): void {
  if (w <= 1 && h <= 1) return;
  const cw = Math.max(1, w >> 1);
  const ch = Math.max(1, h >> 1);
  const coarse = new Float32Array(cw * ch * 4);
  const coarseWeight = new Float32Array(cw * ch);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const wt = weight[i]!;
      if (!wt) continue;
      const o = Math.min(ch - 1, y >> 1) * cw + Math.min(cw - 1, x >> 1);
      for (let c = 0; c < 4; c++) coarse[o * 4 + c]! += color[i * 4 + c]! * wt;
      coarseWeight[o]! += wt;
    }
  for (let o = 0; o < cw * ch; o++)
    if (coarseWeight[o]) {
      for (let c = 0; c < 4; c++) coarse[o * 4 + c]! /= coarseWeight[o]!;
      coarseWeight[o] = Math.min(1, coarseWeight[o]!);
    }
  pushPull(coarse, coarseWeight, cw, ch);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (weight[i]) continue;
      const o = Math.min(ch - 1, y >> 1) * cw + Math.min(cw - 1, x >> 1);
      for (let c = 0; c < 4; c++) color[i * 4 + c] = coarse[o * 4 + c]!;
    }
}
