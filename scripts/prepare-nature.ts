/**
 * Builds the vegetation asset: trees and bushes generated with EZ-Tree (MIT), plus grass,
 * clover, ferns and rocks from Quaternius' Stylized Nature MegaKit (CC0), glTF folder:
 *   bun scripts/prepare-nature.ts <kit>/glTF public/models/nature.glb
 *
 * Every model becomes nodes "<Model>_lod<k>" (k = 0 nearest), one mesh each. Levels of
 * detail come from meshoptimizer (and leaf-card thinning); textures are resized to WebP;
 * meshes meshopt-compressed.
 */
import { readFile } from "node:fs/promises";
import { basename, join } from "node:path";
import {
  Document,
  NodeIO,
  Primitive,
  type Material,
  type Mesh,
  type Texture,
} from "@gltf-transform/core";
import {
  ALL_EXTENSIONS,
  EXTMeshoptCompression,
  KHRMeshQuantization,
} from "@gltf-transform/extensions";
import {
  compactPrimitive,
  prune,
  quantize,
  textureCompress,
  weld,
} from "@gltf-transform/functions";
import { MeshoptEncoder, MeshoptSimplifier } from "meshoptimizer";
import sharp from "sharp";

const [srcDir, outPath] = process.argv.slice(2);
if (!srcDir || !outPath)
  throw new Error("usage: prepare-nature <kit>/glTF <out.glb>");
await MeshoptEncoder.ready;
await MeshoptSimplifier.ready;
const io = new NodeIO()
  .registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ "meshopt.encoder": MeshoptEncoder });

/**
 * Per level of detail: triangle ratio and error (relative to size) for meshes, and the
 * fraction of separate leaf cards kept (each enlarged to keep the crown's coverage).
 */
type Lod = {
  ratio: number;
  error: number;
  cards?: number;
  /** Drop branches whose extent is below this fraction of the model's height. */
  twigs?: number;
  /** Coarse clustering instead of edge collapse (far levels: shape only). */
  sloppy?: boolean;
};
/** Generated trees: heavy (13-24k triangles), so even the nearest level is reduced. */
const EZ_TREE: Lod[] = [
  { ratio: 0.5, error: 0.004, cards: 0.55 },
  { ratio: 0.08, error: 0.05, cards: 0.11, twigs: 0.06 },
];
const EZ_BUSH: Lod[] = [
  { ratio: 0.3, error: 0.01, cards: 0.25 },
  { ratio: 0.08, error: 0.08, cards: 0.05, twigs: 0.15 },
];

/**
 * Trees and bushes generated with EZ-Tree (MIT, Daniel Greenheck): preset, seed, and the
 * height (m) it is normalised to; the game scales them further.
 */
const GENERATED: {
  name: string;
  preset: string;
  seed: number;
  height: number;
  lods: Lod[];
  /** Leaf texture override (the bush presets use autumn aspen and blue-tinted pine). */
  leaves?: string;
}[] = [
  { name: "Oak_1", preset: "Oak Medium", seed: 101, height: 10, lods: EZ_TREE },
  { name: "Oak_2", preset: "Oak Medium", seed: 202, height: 10, lods: EZ_TREE },
  { name: "Oak_3", preset: "Oak Large", seed: 303, height: 12, lods: EZ_TREE },
  { name: "Ash_1", preset: "Ash Medium", seed: 404, height: 11, lods: EZ_TREE },
  { name: "Ash_2", preset: "Ash Medium", seed: 505, height: 11, lods: EZ_TREE },
  {
    name: "Pine_1",
    preset: "Pine Medium",
    seed: 606,
    height: 11,
    lods: EZ_TREE,
  },
  {
    name: "Pine_2",
    preset: "Pine Medium",
    seed: 707,
    height: 11,
    lods: EZ_TREE,
  },
  {
    name: "Pine_3",
    preset: "Pine Large",
    seed: 808,
    height: 13,
    lods: EZ_TREE,
  },
  { name: "Bush_1", preset: "Bush 1", seed: 909, height: 1.6, lods: EZ_BUSH },
  {
    name: "Bush_2",
    preset: "Bush 2",
    seed: 111,
    height: 1.6,
    lods: EZ_BUSH,
    leaves: "oak",
  },
  {
    name: "Bush_3",
    preset: "Bush 3",
    seed: 222,
    height: 1.6,
    lods: EZ_BUSH,
    leaves: "ash",
  },
];
const EZ_ASSETS = "node_modules/@dgreenheck/ez-tree/src/lib/assets";
const TREE: Lod[] = [
  { ratio: 0.5, error: 0.004, cards: 1 },
  { ratio: 0.15, error: 0.02, cards: 0.35 },
  { ratio: 0.03, error: 0.15, cards: 0.05 },
];
const BUSH: Lod[] = [
  { ratio: 1, error: 0 },
  { ratio: 0.25, error: 0.05, cards: 0.35 },
];
const SINGLE: Lod[] = [{ ratio: 1, error: 0 }];

const MODELS: Record<string, Lod[]> = {
  Rock_Medium_1: BUSH,
  Rock_Medium_2: BUSH,
  Rock_Medium_3: BUSH,
  Grass_Common_Short: SINGLE,
  Grass_Common_Tall: SINGLE,
  Grass_Wispy_Short: SINGLE,
  Grass_Wispy_Tall: SINGLE,
  Clover_1: SINGLE,
  Fern_1: SINGLE,
};

/** Source material -> our material, its textures (relative to the kit) and their size. */
const MATERIALS: Record<
  string,
  { name: string; map: string; normal?: string; size: number; mask: boolean }
> = {
  Bark_NormalTree: {
    name: "bark",
    map: "Bark_NormalTree.png",
    normal: "Bark_NormalTree_Normal.png",
    size: 512,
    mask: false,
  },
  Leaves_NormalTree: {
    name: "leaves",
    map: "Leaves_NormalTree.png",
    size: 1024,
    mask: true,
  },
  Leaves_Pine: {
    name: "needles",
    map: "Leaf_Pine.png",
    size: 1024,
    mask: true,
  },
  Leaves_TwistedTree: {
    name: "bush",
    map: "Leaves_TwistedTree.png",
    size: 512,
    mask: true,
  },
  Grass: { name: "grass", map: "Grass.png", size: 256, mask: false },
  Leaves: { name: "plants", map: "Leaves.png", size: 512, mask: true },
  Flowers: { name: "flowers", map: "Flowers.png", size: 512, mask: true },
  Rocks: { name: "rock", map: "Rocks_Diffuse.png", size: 512, mask: false },
};

const out = new Document();
const buffer = out.createBuffer();
const scene = out.createScene();
const textures = new Map<string, Texture>();
const materials = new Map<string, Material>();

async function textureFor(file: string, size: number): Promise<Texture> {
  let t = textures.get(file);
  if (!t) {
    const path = file.includes("/")
      ? file
      : join(srcDir!, "..", "Textures", file);
    const png = await sharp(path)
      .resize(size, size, { fit: "fill" })
      .png()
      .toBuffer();
    t = out
      .createTexture(basename(file, ".png"))
      .setImage(new Uint8Array(png))
      .setMimeType("image/png")
      .setURI(basename(file));
    textures.set(file, t);
  }
  return t;
}

async function materialFor(source: string): Promise<Material> {
  const spec = MATERIALS[source];
  if (!spec) throw new Error(`unmapped material ${source}`);
  let m = materials.get(spec.name);
  if (!m) {
    m = out
      .createMaterial(spec.name)
      .setBaseColorTexture(await textureFor(spec.map, spec.size))
      .setRoughnessFactor(1)
      .setMetallicFactor(0)
      .setDoubleSided(spec.mask)
      .setAlphaMode(spec.mask ? "MASK" : "OPAQUE")
      .setAlphaCutoff(0.2);
    if (spec.normal)
      m.setNormalTexture(await textureFor(spec.normal, spec.size));
    materials.set(spec.name, m);
  }
  return m;
}

// 1. Copy every model's primitives in, with our materials.
const sources = new Map<string, Mesh>();
for (const [model] of Object.entries(MODELS)) {
  const src = await io.read(join(srcDir, `${model}.gltf`));
  const mesh = out.createMesh(model);
  for (const p of src.getRoot().listMeshes()[0]!.listPrimitives()) {
    const prim = out.createPrimitive().setMode(Primitive.Mode.TRIANGLES);
    for (const semantic of ["POSITION", "NORMAL", "TEXCOORD_0", "COLOR_0"]) {
      const a = p.getAttribute(semantic);
      if (!a) continue;
      // Colours to plain floats (some are normalised shorts), alpha dropped.
      let array: Float32Array;
      let type = a.getType();
      if (semantic === "COLOR_0") {
        const e: number[] = [];
        array = new Float32Array(a.getCount() * 3);
        for (let i = 0; i < a.getCount(); i++)
          array.set(a.getElement(i, e).slice(0, 3), i * 3);
        type = "VEC3";
      } else array = new Float32Array(a.getArray()!);
      prim.setAttribute(
        semantic,
        out.createAccessor().setType(type).setArray(array).setBuffer(buffer),
      );
    }
    const idx = p.getIndices()!;
    prim.setIndices(
      out
        .createAccessor()
        .setType("SCALAR")
        .setArray(new Uint32Array(idx.getArray()!))
        .setBuffer(buffer),
    );
    prim.setMaterial(await materialFor(p.getMaterial()!.getName()));
    mesh.addPrimitive(prim);
  }
  sources.set(model, mesh);
}
// Generated trees: geometry only (three's texture loading is stubbed out).
const stub = () => ({
  addEventListener() {},
  removeEventListener() {},
  setAttribute() {},
  style: {},
});
(globalThis as { document?: unknown }).document = {
  createElementNS: stub,
  createElement: stub,
};
const { Tree } = await import("@dgreenheck/ez-tree");
for (const g of GENERATED) {
  const tree = new Tree();
  tree.loadPreset(g.preset);
  tree.options.seed = g.seed;
  if (g.leaves)
    (tree.options as unknown as { leaves: { type: string } }).leaves.type =
      g.leaves;
  tree.generate();
  const opts = tree.options as unknown as {
    bark: { type: string; textureScale: { x: number; y: number } };
    leaves: { type: string };
  };
  const parts = [tree.branchesMesh.geometry, tree.leavesMesh.geometry];
  parts.forEach((p) => p.computeBoundingBox());
  const height = Math.max(...parts.map((p) => p.boundingBox!.max.y));
  const scale = g.height / height;
  const mesh = out.createMesh(g.name);
  for (const [k, geo] of parts.entries()) {
    const bark = k === 0;
    const pos = new Float32Array(geo.getAttribute("position").array).map(
      (v) => v * scale,
    );
    const uv = new Float32Array(geo.getAttribute("uv").array);
    // EZ-Tree tiles the bark with a texture repeat; bake it into the UVs.
    if (bark)
      for (let i = 0; i < uv.length; i += 2) {
        uv[i] *= opts.bark.textureScale.x;
        uv[i + 1] /= opts.bark.textureScale.y;
      }
    const normal = new Float32Array(geo.getAttribute("normal").array);
    if (!bark) crownNormals(pos, normal);
    const acc = (type: "VEC2" | "VEC3", a: Float32Array) =>
      out.createAccessor().setType(type).setArray(a).setBuffer(buffer);
    const prim = out
      .createPrimitive()
      .setAttribute("POSITION", acc("VEC3", pos))
      .setAttribute("NORMAL", acc("VEC3", normal))
      .setAttribute("TEXCOORD_0", acc("VEC2", uv))
      .setIndices(
        out
          .createAccessor()
          .setType("SCALAR")
          .setArray(new Uint32Array(geo.index!.array))
          .setBuffer(buffer),
      )
      .setMaterial(
        await generatedMaterial(
          bark ? `bark_${opts.bark.type}` : `leaves_${opts.leaves.type}`,
          bark,
        ),
      );
    mesh.addPrimitive(prim);
  }
  sources.set(g.name, mesh);
  MODELS[g.name] = g.lods;
}

/**
 * Leaf normals bent outwards from the crown's centre (and a little up), so the crown shades
 * as one soft volume instead of thousands of flat cards facing every way.
 */
function crownNormals(pos: Float32Array, normal: Float32Array): void {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i++) {
    min[i % 3] = Math.min(min[i % 3]!, pos[i]!);
    max[i % 3] = Math.max(max[i % 3]!, pos[i]!);
  }
  const c = min.map((v, k) => (v + max[k]!) / 2);
  const half = max.map((v, k) => (v - min[k]!) / 2 || 1);
  for (let i = 0; i < pos.length; i += 3) {
    const d = [0, 1, 2].map((k) => (pos[i + k]! - c[k]!) / half[k]!);
    d[1]! += 0.3;
    const l = Math.hypot(...d) || 1;
    const n = [0, 1, 2].map((k) => 0.25 * normal[i + k]! + 0.75 * (d[k]! / l));
    const ln = Math.hypot(...n) || 1;
    for (let k = 0; k < 3; k++) normal[i + k] = n[k]! / ln;
  }
}

async function generatedMaterial(
  name: string,
  bark: boolean,
): Promise<Material> {
  let m = materials.get(name);
  if (!m) {
    const type = name.split("_")[1]!;
    m = out
      .createMaterial(name)
      .setRoughnessFactor(1)
      .setMetallicFactor(0)
      .setDoubleSided(!bark)
      .setAlphaMode(bark ? "OPAQUE" : "MASK")
      .setAlphaCutoff(0.5)
      .setBaseColorTexture(
        await textureFor(
          bark
            ? `${EZ_ASSETS}/bark/${type}_color_1k.jpg`
            : `${EZ_ASSETS}/leaves/${type}_color.png`,
          512,
        ),
      );
    if (bark)
      m.setNormalTexture(
        await textureFor(`${EZ_ASSETS}/bark/${type}_normal_1k.jpg`, 512),
      );
    materials.set(name, m);
  }
  return m;
}

await out.transform(weld());

// 2. Levels of detail, as separate meshes and nodes.
let report = "";
for (const [model, lods] of Object.entries(MODELS)) {
  const base = sources.get(model)!;
  const counts: number[] = [];
  lods.forEach((lod, k) => {
    // Own primitives and vertex data per level (Mesh.clone() shares them): each level is
    // simplified from the original, and quantization stores its scale on one node per mesh.
    const mesh = out.createMesh();
    for (const src of base.listPrimitives()) {
      const prim = src.clone();
      for (const semantic of prim.listSemantics())
        prim.setAttribute(semantic, prim.getAttribute(semantic)!.clone());
      mesh.addPrimitive(prim);
    }
    mesh.setName(`${model}_lod${k}`);
    let tris = 0;
    for (const prim of mesh.listPrimitives()) {
      if (lod.ratio < 1) simplifyPrimitive(prim, lod, model);
      tris += prim.getIndices()!.getCount() / 3;
    }
    counts.push(tris);
    scene.addChild(out.createNode(`${model}_lod${k}`).setMesh(mesh));
  });
  report += `${model}: ${counts.join(" / ")}\n`;
}

function simplifyPrimitive(prim: Primitive, lod: Lod, model: string): void {
  const indices = new Uint32Array(prim.getIndices()!.getArray()!);
  const pos = new Float32Array(prim.getAttribute("POSITION")!.getArray()!);
  const components = connectedComponents(indices, pos.length / 3);
  let dst: Uint32Array;
  if (components.count * 2.5 > indices.length / 3) {
    // Loose leaf cards: thin them out and enlarge the rest.
    dst = thinCards(prim, indices, pos, components.of, lod.cards ?? 1, model);
  } else {
    if (lod.twigs) dropTwigs(indices, pos, components.of, lod.twigs);
    // Keep the shading (normals) and texture mapping (UVs) as well as the shape. Permissive:
    // the bark's UV and colour seams would otherwise lock nearly every vertex.
    const target0 = Math.max(
      3,
      Math.floor((lod.ratio * indices.length) / 3) * 3,
    );
    if (lod.sloppy) {
      [dst] = MeshoptSimplifier.simplifySloppy(
        indices,
        pos,
        3,
        null,
        target0,
        lod.error,
      );
      prim.setIndices(
        out.createAccessor().setType("SCALAR").setArray(dst).setBuffer(buffer),
      );
      compactPrimitive(prim);
      return;
    }
    const nor = prim.getAttribute("NORMAL")!.getArray()!;
    const uv = prim.getAttribute("TEXCOORD_0")!.getArray()!;
    const n = pos.length / 3;
    const attrs = new Float32Array(n * 5);
    for (let i = 0; i < n; i++) {
      attrs.set([nor[i * 3]!, nor[i * 3 + 1]!, nor[i * 3 + 2]!], i * 5);
      attrs.set([uv[i * 2]!, uv[i * 2 + 1]!], i * 5 + 3);
    }
    const target = Math.max(
      3,
      Math.floor((lod.ratio * indices.length) / 3) * 3,
    );
    [dst] = MeshoptSimplifier.simplifyWithAttributes(
      indices,
      pos,
      3,
      attrs,
      5,
      [0.5, 0.5, 0.5, 0.5, 0.5],
      null,
      target,
      lod.error,
      ["Permissive"],
    );
  }
  prim.setIndices(
    out.createAccessor().setType("SCALAR").setArray(dst).setBuffer(buffer),
  );
  compactPrimitive(prim);
}

/**
 * Removes (in place, as degenerate triangles) separate branches whose extent is below `fraction`
 * of the model's height.
 */
function dropTwigs(
  indices: Uint32Array,
  pos: Float32Array,
  of: Int32Array,
  fraction: number,
): void {
  let top = 0;
  for (let i = 1; i < pos.length; i += 3) top = Math.max(top, pos[i]!);
  const box = new Map<number, number[]>();
  for (let v = 0; v < pos.length / 3; v++) {
    const b = box.get(of[v]!) ?? [
      Infinity,
      Infinity,
      Infinity,
      -Infinity,
      -Infinity,
      -Infinity,
    ];
    for (let k = 0; k < 3; k++) {
      b[k] = Math.min(b[k]!, pos[v * 3 + k]!);
      b[k + 3] = Math.max(b[k + 3]!, pos[v * 3 + k]!);
    }
    box.set(of[v]!, b);
  }
  const small = (c: number) => {
    const b = box.get(c)!;
    return (
      Math.hypot(b[3]! - b[0]!, b[4]! - b[1]!, b[5]! - b[2]!) < fraction * top
    );
  };
  for (let i = 0; i < indices.length; i += 3)
    if (small(of[indices[i]!]!)) indices[i + 1] = indices[i + 2] = indices[i]!;
}

/** Component id of every vertex, by triangles sharing vertices. */
function connectedComponents(
  indices: Uint32Array,
  vertices: number,
): { of: Int32Array; count: number } {
  const parent = new Int32Array(vertices).map((_, i) => i);
  const find = (x: number): number => {
    while (parent[x] !== x) x = parent[x] = parent[parent[x]!]!;
    return x;
  };
  for (let i = 0; i < indices.length; i += 3) {
    const a = find(indices[i]!);
    parent[find(indices[i + 1]!)] = a;
    parent[find(indices[i + 2]!)] = a;
  }
  const of = new Int32Array(vertices).map((_, i) => find(i));
  return { of, count: new Set(indices.map((i) => of[i]!)).size };
}

/**
 * Keeps a deterministic `keep` fraction of the separate cards, each scaled about its centre by
 * 1/sqrt(keep) so the crown covers about as much; positions are written to a new accessor.
 */
function thinCards(
  prim: Primitive,
  indices: Uint32Array,
  pos: Float32Array,
  of: Int32Array,
  keep: number,
  model: string,
): Uint32Array {
  if (keep >= 1) return indices;
  const seed = [...model].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);
  const kept = (c: number) =>
    (((c * 2654435761) ^ seed) >>> 0) / 2 ** 32 < keep;
  const centre = new Map<number, [number, number, number, number]>();
  for (let v = 0; v < pos.length / 3; v++) {
    const c = centre.get(of[v]!) ?? [0, 0, 0, 0];
    c[0] += pos[v * 3]!;
    c[1] += pos[v * 3 + 1]!;
    c[2] += pos[v * 3 + 2]!;
    c[3]++;
    centre.set(of[v]!, c);
  }
  const scale = 1 / Math.sqrt(keep);
  const moved = new Float32Array(pos);
  for (let v = 0; v < pos.length / 3; v++) {
    const c = centre.get(of[v]!)!;
    for (let k = 0; k < 3; k++)
      moved[v * 3 + k] =
        c[k]! / c[3]! + (pos[v * 3 + k]! - c[k]! / c[3]!) * scale;
  }
  prim.setAttribute(
    "POSITION",
    out.createAccessor().setType("VEC3").setArray(moved).setBuffer(buffer),
  );
  const outIdx: number[] = [];
  for (let i = 0; i < indices.length; i += 3)
    if (kept(of[indices[i]!]!))
      outIdx.push(indices[i]!, indices[i + 1]!, indices[i + 2]!);
  return new Uint32Array(outIdx);
}

for (const m of sources.values())
  if (!m.listParents().some((p) => p !== out.getRoot())) m.dispose();
await out.transform(
  prune(),
  textureCompress({ encoder: sharp, targetFormat: "webp" }),
  quantize(),
);
out
  .createExtension(EXTMeshoptCompression)
  .setRequired(true)
  .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
out.createExtension(KHRMeshQuantization).setRequired(true);
await io.write(outPath, out);
console.log(report.trim());
console.log(`written ${outPath}`);
