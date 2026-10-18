/**
 * Turns the downloaded BMW i4 GLB into a game-ready asset:
 *   bun scripts/prepare-car.ts <source.glb> public/models/bmw-i4.glb
 *
 * - drops the interior and the licence plate
 * - bakes all transforms, scales to metres (by overall length) and turns the car to face -Z
 * - splits wheel geometry per corner into a spinning part (tyre, rim, disc, nuts) and a
 *   steering-only part (caliper), each centred on its wheel centre
 * - merges the body by material, simplifies, resizes textures and meshopt-compresses
 *
 * Output: nodes "body", "wheel_<fl|fr|rl|rr>_spin" and "wheel_<..>_caliper"; origin on the
 * ground, midway between the axles; extras.wheels holds the measured wheel centres/radius.
 */
import { NodeIO, type Primitive } from "@gltf-transform/core";
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
  simplify,
  textureCompress,
  transformPrimitive,
  weld,
  quantize,
} from "@gltf-transform/functions";
import { MeshoptEncoder, MeshoptSimplifier } from "meshoptimizer";

const [srcPath, outPath] = process.argv.slice(2);
if (!srcPath || !outPath)
  throw new Error("usage: prepare-car.ts <source.glb> <out.glb>");
const LENGTH = 4.783; // m, BMW i4
const DROP = [/Interior/, /License_Plate/];
const STATIC = /Calipers/;

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

const allPrims = () =>
  root
    .listMeshes()
    .flatMap((mesh) => mesh.listPrimitives().map((prim) => ({ mesh, prim })));
for (const { mesh, prim } of allPrims())
  if (DROP.some((re) => re.test(prim.getMaterial()?.getName() ?? ""))) {
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
const tyres = allPrims().filter(({ prim }) =>
  /Tires/.test(prim.getMaterial()?.getName() ?? ""),
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
  return r < 0.97 * w.radius && lat > -0.03 && lat < width + 0.04 ? c : null;
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
for (const { prim } of allPrims()) {
  const pos = prim.getAttribute("POSITION")!;
  const idx = prim.getIndices();
  const count = (idx ? idx.getCount() : pos.getCount()) / 3;
  const buckets = new Map<string, number[]>();
  const a = [0, 0, 0];
  const b = [0, 0, 0];
  const c = [0, 0, 0];
  const isStatic = STATIC.test(prim.getMaterial()?.getName() ?? "");
  for (let t = 0; t < count; t++) {
    pos.getElement(idx ? idx.getScalar(t * 3) : t * 3, a);
    pos.getElement(idx ? idx.getScalar(t * 3 + 1) : t * 3 + 1, b);
    pos.getElement(idx ? idx.getScalar(t * 3 + 2) : t * 3 + 2, c);
    const x = (a[0]! + b[0]! + c[0]!) / 3;
    const y = (a[1]! + b[1]! + c[1]!) / 3;
    const z = (a[2]! + b[2]! + c[2]!) / 3;
    const w = isStatic ? cornerOf(x, z) : wheelOf(x, y, z);
    const key = w ? `wheel_${w}_${isStatic ? "caliper" : "spin"}` : "body";
    buckets.set(key, [...(buckets.get(key) ?? []), t]);
  }
  for (const [key, tris] of buckets) add(key, extract(prim, tris));
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

const before = triangles();
await doc.transform(
  dedup(),
  join({ keepNamed: true }),
  // The source stores nearly (not bitwise) identical vertices per triangle; snapping them to a
  // quantization grid first lets weld merge them, which simplify needs.
  quantize({ quantizePosition: 14, quantizeNormal: 8, quantizeTexcoord: 12 }),
  dequantize(),
  weld(),
  simplify({ simplifier: MeshoptSimplifier, ratio: 0.25, error: 0.003 }),
  textureCompress({ targetFormat: "webp", resize: [1024, 1024] }),
  prune(),
  resample(),
  quantize(),
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
