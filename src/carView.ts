import * as THREE from "three/webgpu";
import {
  attribute,
  cameraPosition,
  mix,
  normalWorld,
  pmremTexture,
  positionWorld,
  reflect,
  smoothstep,
  step,
  uniform,
  vec3,
} from "three/tsl";
import type { Node, UniformNode } from "three/webgpu";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import type { Car } from "./sim/car";
import { trackDownload } from "./loading";

/** Reflected land below the horizon, relative to the sky light there. */
const GROUND_REFLECTION = vec3(0.1, 0.11, 0.09);
/** Lamp emission strengths (scene-linear; bloom starts around 1.6). */
const DRL = 2.5;
const REVERSE = 2.5;
/** Headlight intensity (candela-like), per lamp. */
const HEADLIGHT = 600;
const TAIL = 2;
const BRAKE = 25;
const LAMP_WHITE = vec3(1, 0.97, 0.92);
const LAMP_RED = vec3(1, 0.03, 0.01);

/** Emission levels the lamp materials read, set every frame. */
interface Lamps {
  /** Front lamps: daytime running lights, brighter as headlights at night. */
  front: UniformNode<"float", number>;
  /** Tail lamps: always on, full when braking. */
  tail: UniformNode<"float", number>;
  /** Brake-only: the third brake light. */
  brake: UniformNode<"float", number>;
  reverse: UniformNode<"float", number>;
}

const CORNERS = ["fl", "fr", "rl", "rr"] as const;
/** Model parts that cast into the shadow map; the small ones (badges, lamps, rotors) don't. */
const SHADOW_CASTERS = /^plain|Tires|White|Paint|_Windows$/;
const normal = new THREE.Vector3();
const forward = new THREE.Vector3();
const side = new THREE.Vector3();
const basis = new THREE.Matrix4();

/** A ground-aligned quad with a soft radial falloff, sized to the car's footprint. */
function contactShadow(car: Car): THREE.Mesh {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createRadialGradient(
    size / 2,
    size / 2,
    0,
    size / 2,
    size / 2,
    size / 2,
  );
  gradient.addColorStop(0, "rgba(0,0,0,1)");
  gradient.addColorStop(0.55, "rgba(0,0,0,0.85)");
  gradient.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  const [, , hz] = car.cfg.body.halfExtents;
  const [hx] = car.cfg.body.halfExtents;
  const geometry = new THREE.PlaneGeometry(2 * hx + 0.5, 2 * hz + 0.6).rotateX(
    -Math.PI / 2,
  );
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    color: 0x000000,
  });
  // Drawn on top of the ground but never over the car.
  material.polygonOffset = true;
  material.polygonOffsetFactor = -2;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = 1;
  return mesh;
}
const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);

const box = (w: number, h: number, d: number, color: number, emissive = 0) => {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, d),
    new THREE.MeshStandardMaterial({
      color,
      emissive,
      flatShading: true,
      roughness: 0.6,
    }),
  );
  mesh.castShadow = true;
  return mesh;
};

/** Low-poly car; all geometry is relative to the centre of mass. */
export class CarView {
  /** Add this to the scene: the car plus its contact shadow on the ground. */
  readonly group = new THREE.Group();
  /** The car itself, moved with the physics body. */
  readonly root = new THREE.Group();
  /** Resolves once the model (if any) has replaced the placeholder. */
  readonly ready: Promise<void>;
  /**
   * A soft dark patch on the ground under the car. Shadow maps are too coarse for the
   * contact shading right under the sills and tyres; without it the car looks like it floats.
   */
  private readonly contact: THREE.Mesh;
  private readonly wheels: { steer: THREE.Group; spin: THREE.Group }[] = [];
  private readonly envMaterials: THREE.MeshStandardMaterial[] = [];
  private readonly lamps: Lamps = {
    front: uniform(DRL),
    tail: uniform(TAIL),
    brake: uniform(0),
    reverse: uniform(0),
  };
  /** Boxes shown until (or instead of) a model. */
  private readonly placeholder = new THREE.Group();
  /** Low beams, lit at dusk and night. No shadows, to stay cheap. */
  private readonly headlights: THREE.SpotLight[] = [];

  constructor(
    private readonly car: Car,
    paint = 0xd2552d,
    /** Optional GLB with nodes "body", "wheel_<fl|fr|rl|rr>_spin" and "wheel_<..>_caliper". */
    model?: { url: string; envMap?: THREE.Texture },
  ) {
    this.root.add(this.placeholder);
    this.contact = contactShadow(car);
    this.group.add(this.root, this.contact);
    // Lamp positions in the model (axle-midpoint origin), moved to centre-of-mass coordinates.
    const mid = (car.cfg.front.z + car.cfg.rear.z) / 2;
    for (const x of [-0.7, 0.7]) {
      const light = new THREE.SpotLight(0xfff4e6, 0, 140, Math.PI / 7, 0.6, 2);
      light.position.set(x, 0.72 - car.cfg.cgHeight, -2.05 + mid);
      // Aimed slightly down, to meet the road about 40 m ahead.
      light.target.position.set(x * 2, -car.cfg.cgHeight, -40 + mid);
      this.root.add(light, light.target);
      this.headlights.push(light);
    }
    // Proportions follow the collision box, so each car's placeholder matches its size.
    const [, cy, cz] = car.cfg.body.center;
    const [hx, hy, hz] = car.cfg.body.halfExtents;
    const top = cy + hy;
    const bottom = cy - hy;
    const width = 2 * hx - 0.12; // let the wheels show past the body sides
    const length = 2 * hz;
    const beltline = bottom + (top - bottom) * 0.55;
    const lower = box(width, beltline - bottom, length, paint);
    lower.position.set(0, (bottom + beltline) / 2, cz);
    const cabin = box(width * 0.84, top - beltline, length * 0.46, 0x2b2f36);
    cabin.position.set(0, (beltline + top) / 2, cz + length * 0.06);
    const nose = box(width * 0.92, 0.08, length * 0.26, paint);
    nose.position.set(0, beltline + 0.04, cz - length * 0.33);
    this.placeholder.add(lower, cabin, nose);
    const front = cz - hz;
    const rear = cz + hz;
    for (const x of [-width * 0.38, width * 0.38]) {
      const head = box(width * 0.2, 0.1, 0.05, 0xffffff, 0xfff2cc);
      head.position.set(x, beltline - 0.12, front - 0.02);
      const tail = box(width * 0.22, 0.08, 0.05, 0x550000, 0xaa0000);
      tail.position.set(x, beltline - 0.06, rear + 0.02);
      this.placeholder.add(head, tail);
    }

    const tireMat = new THREE.MeshStandardMaterial({
      color: 0x1c1c1c,
      roughness: 0.9,
      flatShading: true,
    });
    const rimMat = new THREE.MeshStandardMaterial({
      color: 0xb8bcc4,
      metalness: 0.6,
      roughness: 0.4,
    });
    for (const w of car.wheels) {
      const steer = new THREE.Group();
      const spin = new THREE.Group();
      const r = w.axle.wheelRadius;
      const tire = new THREE.Mesh(
        new THREE.CylinderGeometry(r, r, 0.24, 14),
        tireMat,
      );
      tire.rotation.z = Math.PI / 2;
      tire.castShadow = true;
      const spoke = new THREE.Mesh(
        new THREE.BoxGeometry(0.02, r * 1.5, 0.12),
        rimMat,
      );
      spoke.position.x = (w.side * 0.24) / 2;
      const spoke2 = spoke.clone();
      spoke2.rotation.x = Math.PI / 2;
      const parts = new THREE.Group();
      parts.name = "placeholder";
      parts.add(tire, spoke, spoke2);
      spin.add(parts);
      steer.add(spin);
      steer.position.copy(w.mount);
      this.root.add(steer);
      this.wheels.push({ steer, spin });
    }
    this.ready = model ? this.load(model.url, model.envMap) : Promise.resolve();
  }

  /**
   * Swaps the placeholder for a model. Its origin is on the ground midway between the axles;
   * wheel parts are centred on their wheel centres and scaled to the physics wheel radius.
   */
  private async load(url: string, envMap?: THREE.Texture): Promise<void> {
    const gltf = await loader.loadAsync(url, trackDownload(url));
    const scene = gltf.scene;
    const info = scene.userData as {
      radius: number;
      wheels: Record<(typeof CORNERS)[number], { center: number[] }>;
    };
    scene.updateMatrixWorld(true);
    scene.traverse((o) => {
      if (!(o instanceof THREE.Mesh)) return;
      o.castShadow = true;
      o.receiveShadow = true;
      const name = (o.material as THREE.Material).name;
      // Lamp parts that share a material, told apart by where they are on the car.
      if (/DRLs$/.test(name)) mark(o, "front", (p) => (p.z < 0 ? 1 : 0));
      if (/_Windows$/.test(name)) mark(o, "carPos", (p) => p.toArray(), 3);
      if (/DarkRed$/.test(name))
        mark(o, "zone", (p) => (p.y > 1.15 ? 2 : p.y > 0.65 ? 1 : 0));
      o.material = upgrade(o.material as THREE.Material, this.lamps, envMap);
      const m = o.material as THREE.MeshStandardMaterial;
      if ("envMapIntensity" in m && m.envMap) this.envMaterials.push(m);
    });
    // Nodes carry their own (dequantization) transforms, so each is wrapped rather than moved.
    const cfg = this.car.cfg;
    const body = new THREE.Group();
    body.add(scene.getObjectByName("body")!);
    body.position.set(0, -cfg.cgHeight, (cfg.front.z + cfg.rear.z) / 2);
    this.root.add(body);
    this.root.remove(this.placeholder);
    CORNERS.forEach((corner, i) => {
      const wheel = this.wheels[i]!;
      const [x, y, z] = info.wheels[corner].center as [number, number, number];
      const scale = this.car.wheels[i]!.axle.wheelRadius / info.radius;
      wheel.spin.remove(wheel.spin.getObjectByName("placeholder")!);
      for (const [part, parent] of [
        ["spin", wheel.spin],
        ["caliper", wheel.steer],
      ] as const) {
        const node = scene.getObjectByName(`wheel_${corner}_${part}`);
        if (!node) continue;
        // Re-centre on the wheel centre, keeping the node's own transform.
        node.position.sub(new THREE.Vector3(x, y, z));
        const holder = new THREE.Group();
        holder.scale.setScalar(scale);
        holder.add(node);
        parent.add(holder);
      }
    });
    this.mergeParts(body);
  }

  /**
   * Fewer draw calls (the model's ~50 meshes cost more CPU than the rest of the car): parts
   * sharing a material become one skinned mesh across the body and all four wheels, each part
   * bound rigidly to a bone (the body, its wheel's spin or, for calipers, steering group).
   * Plain parts, untextured and differing only in colour, share one material with the
   * colour per vertex.
   */
  private mergeParts(body: THREE.Object3D): void {
    const root = this.root;
    root.updateMatrixWorld(true);
    const bones = [body, ...this.wheels.flatMap((w) => [w.spin, w.steer])];
    const boneOf = (o: THREE.Object3D) => {
      for (let p: THREE.Object3D | null = o; p; p = p.parent) {
        const i = bones.indexOf(p);
        if (i >= 0) return i;
      }
      throw new Error("car part outside the body and wheels");
    };
    const groups = new Map<string, THREE.Mesh[]>();
    const keyOf = new Map<THREE.Material, string>();
    root.traverse((o) => {
      if (!(o instanceof THREE.Mesh)) return;
      const m = o.material as THREE.MeshStandardMaterial;
      const plain =
        m.type === "MeshStandardMaterial" &&
        !m.transparent &&
        !m.map &&
        !m.normalMap &&
        !m.emissiveMap &&
        m.emissive.getHex() === 0;
      let key = keyOf.get(m);
      if (!key) {
        key = plain
          ? `plain ${m.roughness} ${m.metalness} ${m.side}`
          : `${m.name} ${m.uuid}`;
        keyOf.set(m, key);
      }
      const list = groups.get(key);
      if (list) list.push(o);
      else groups.set(key, [o]);
    });
    const skeleton = new THREE.Skeleton(bones as THREE.Bone[]);
    const toRoot = new THREE.Matrix4();
    const rootInverse = root.matrixWorld.clone().invert();
    for (const [key, meshes] of groups) {
      if (meshes.length < 2) continue;
      const plain = key.startsWith("plain");
      const geos = meshes.map((mesh) => {
        const g = floatGeometry(mesh.geometry);
        g.applyMatrix4(toRoot.multiplyMatrices(rootInverse, mesh.matrixWorld));
        const n = g.getAttribute("position").count;
        const index = new Uint16Array(n * 4);
        const weight = new Float32Array(n * 4);
        const bone = boneOf(mesh);
        for (let i = 0; i < n; i++) {
          index[i * 4] = bone;
          weight[i * 4] = 1;
        }
        g.setAttribute("skinIndex", new THREE.BufferAttribute(index, 4));
        g.setAttribute("skinWeight", new THREE.BufferAttribute(weight, 4));
        if (plain) {
          const c = (mesh.material as THREE.MeshStandardMaterial).color;
          const color = new Float32Array(n * 3);
          for (let i = 0; i < n; i++) c.toArray(color, i * 3);
          g.setAttribute("color", new THREE.BufferAttribute(color, 3));
        }
        return g;
      });
      // Only attributes every part has (some plain parts carry unused UVs).
      const names = Object.keys(geos[0]!.attributes).filter((a) =>
        geos.every((g) => g.hasAttribute(a)),
      );
      for (const g of geos)
        for (const a of Object.keys(g.attributes))
          if (!names.includes(a)) g.deleteAttribute(a);
      const source = meshes[0]!.material as THREE.MeshStandardMaterial;
      let material: THREE.Material = source;
      if (plain) {
        const m = new THREE.MeshStandardMaterial({
          name: "plain",
          vertexColors: true,
          roughness: source.roughness,
          metalness: source.metalness,
          side: source.side,
          envMap: source.envMap,
          envMapIntensity: source.envMapIntensity,
        });
        if (m.envMap) this.envMaterials.push(m);
        material = m;
      }
      const merged = new THREE.SkinnedMesh(mergeGeometries(geos)!, material);
      merged.name = plain ? "plain" : source.name;
      // It always contains the car, which the camera always looks at.
      merged.frustumCulled = false;
      root.add(merged);
      merged.updateMatrixWorld(true);
      merged.bind(skeleton, merged.matrixWorld);
      for (const mesh of meshes) {
        mesh.removeFromParent();
        mesh.geometry.dispose();
      }
      for (const g of geos) g.dispose();
    }
    root.traverse((o) => {
      if (!(o instanceof THREE.Mesh)) return;
      o.castShadow = SHADOW_CASTERS.test((o.material as THREE.Material).name);
      o.receiveShadow = true;
    });
  }

  /** Headlights and lamp glow for how dark it is (0 day .. 1 night). */
  setHeadlights(darkness: number): void {
    const on = THREE.MathUtils.smoothstep(darkness, 0.05, 0.35);
    for (const light of this.headlights) light.intensity = HEADLIGHT * on;
    this.lamps.front.value = DRL * (1 + 1.5 * on);
  }

  /** Strength of the sky light and reflections on the car's materials. */
  setEnvironmentIntensity(value: number): void {
    for (const m of this.envMaterials)
      m.envMapIntensity = value * ((m.userData.envScale as number) ?? 1);
  }

  /** `alpha` blends between the previous and current physics state. */
  update(alpha: number): void {
    const c = this.car;
    const braking = c.brake > 0.05 ? 1 : 0;
    this.lamps.tail.value = TAIL + (BRAKE - TAIL) * braking;
    this.lamps.brake.value = 0.5 * BRAKE * braking;
    this.lamps.reverse.value = c.gear < 0 ? REVERSE : 0;
    this.root.position.lerpVectors(c.prevPosition, c.position, alpha);
    this.root.quaternion.slerpQuaternions(
      c.prevOrientation,
      c.orientation,
      alpha,
    );
    // Contact shadow: on the ground below the car, turned with it and tilted to the slope,
    // fading as the car leaves the ground.
    const p = this.root.position;
    const ground = c.ground.height(p.x, p.z);
    c.ground.normal(p.x, p.z, normal);
    forward.set(0, 0, -1).applyQuaternion(this.root.quaternion);
    forward.addScaledVector(normal, -forward.dot(normal)).normalize();
    side.crossVectors(forward, normal);
    basis.makeBasis(side, normal, forward.negate());
    this.contact.quaternion.setFromRotationMatrix(basis);
    this.contact.position.set(p.x, ground + 0.03, p.z);
    const height = p.y - ground - c.cfg.cgHeight;
    (this.contact.material as THREE.MeshBasicMaterial).opacity =
      0.75 * THREE.MathUtils.clamp(1 - height / 1.2, 0, 1);
    c.wheels.forEach((w, i) => {
      const v = this.wheels[i]!;
      v.steer.position.y = w.mount.y - w.length;
      v.steer.rotation.y = -w.steer;
      v.spin.rotation.x = -w.spin;
    });
  }
}

/**
 * Car paint gets a clear coat (a glossy layer over the base colour) and glass a mirror-smooth
 * surface (the windows tinted); the source materials are plain and read as matte plastic.
 */
/** Adds a per-vertex float attribute computed from the vertex's position on the car. */
function mark(
  mesh: THREE.Mesh,
  name: string,
  value: (p: THREE.Vector3) => number | number[],
  size = 1,
): void {
  const pos = mesh.geometry.getAttribute("position");
  const out = new Float32Array(pos.count * size);
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++)
    out.set(
      [
        value(v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld)),
      ].flat(),
      i * size,
    );
  mesh.geometry.setAttribute(name, new THREE.BufferAttribute(out, size));
}

/** A copy of the geometry with float attributes (the model's are quantized) and an index. */
function floatGeometry(src: THREE.BufferGeometry): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  let count = 0;
  for (const [name, a] of Object.entries(src.attributes)) {
    const size = a.itemSize;
    const get = [a.getX, a.getY, a.getZ, a.getW].slice(0, size);
    const out = new Float32Array(a.count * size);
    for (let i = 0; i < a.count; i++)
      for (let c = 0; c < size; c++) out[i * size + c] = get[c]!.call(a, i);
    g.setAttribute(name, new THREE.BufferAttribute(out, size));
    count = a.count;
  }
  g.setIndex(
    src.index
      ? Array.from(src.index.array)
      : Array.from({ length: count }, (_, i) => i),
  );
  return g;
}

/** Emissive version of a lamp material, glowing `color` times `level`. */
function lamp(
  material: THREE.MeshStandardMaterial,
  emissive: Node<"vec3">,
): THREE.Material {
  const m = new THREE.MeshStandardNodeMaterial({
    name: material.name,
    color: material.color,
    roughness: 0.3,
    metalness: 0,
    side: material.side,
  });
  m.emissiveNode = emissive;
  material.dispose();
  return m;
}

function upgrade(
  material: THREE.Material,
  lamps: Lamps,
  envMap?: THREE.Texture,
): THREE.Material {
  if (!(material instanceof THREE.MeshStandardMaterial)) return material;
  if (envMap) material.envMap = envMap;
  // Daytime running lights at the front; the same material at the rear is the reversing lights.
  if (/DRLs$/.test(material.name))
    return lamp(
      material,
      LAMP_WHITE.mul(
        mix(lamps.reverse, lamps.front, attribute<"float">("front", "float")),
      ),
    );
  if (/RedMain$/.test(material.name))
    return lamp(material, LAMP_RED.mul(lamps.tail));
  if (/DarkRed$/.test(material.name)) {
    // Zone 2 is the third brake light, 1 the tail lamp clusters around the light guides
    // (a faint glow), 0 the unlit bumper lamps.
    const zone = attribute<"float">("zone", "float");
    return lamp(
      material,
      LAMP_RED.mul(
        mix(lamps.tail.mul(0.08), lamps.brake, zone.sub(1).clamp()).mul(
          zone.clamp(),
        ),
      ),
    );
  }
  if (/BrakeLightsGlass$/.test(material.name)) {
    // Clear red lens rather than the source's dark smoked one, which hid the lamps.
    material.color.setRGB(0.25, 0.01, 0.01);
    material.opacity = 0.25;
    material.roughness = 0.05;
  }
  if (/White|Paint/i.test(material.name)) {
    const paint = new THREE.MeshPhysicalMaterial({
      name: material.name,
      // Pure white clips under the sun; real white paint reflects roughly 70-80%.
      color: material.color.clone().multiplyScalar(0.72),
      roughness: 0.35,
      metalness: 0,
      clearcoat: 1,
      clearcoatRoughness: 0.04,
      envMap: material.envMap,
      envMapIntensity: material.envMapIntensity,
    });
    material.dispose();
    return paint;
  }
  if (envMap && /_Windows$/i.test(material.name)) {
    // Dark tinted privacy glass, opaque since there is no interior to see. A second glossy
    // layer stands in for the pane's back surface.
    const glass = new THREE.MeshPhysicalNodeMaterial({
      name: material.name,
      color: new THREE.Color(0.006, 0.008, 0.01),
      roughness: 0,
      metalness: 0,
      ior: 1.52,
      clearcoat: 1,
      clearcoatRoughness: 0,
      side: material.side,
      envMap,
      envMapIntensity: material.envMapIntensity,
    });
    // The sky light has no ground in it, so below the horizon the reflection would show sky
    // and read milky; darken it to the land's tone instead.
    const reflected = reflect(
      positionWorld.sub(cameraPosition).normalize(),
      normalWorld,
    );
    glass.envNode = pmremTexture(envMap).mul(
      mix(GROUND_REFLECTION, vec3(1), smoothstep(-0.04, 0.06, reflected.y)),
    );
    // The third brake light sits behind the rear window, where the opaque glass would hide
    // it; the glass glows over it instead (the light strip spans |x| < 0.18 m, y 1.314-1.331 m).
    const p = attribute<"vec3">("carPos", "vec3");
    // A thin band just above the strip's own height, since the glass slopes back over it.
    const strip = smoothstep(0.205, 0.195, p.x.abs())
      .mul(smoothstep(1.325, 1.327, p.y))
      .mul(smoothstep(1.335, 1.333, p.y))
      .mul(step(1.3, p.z));
    glass.emissiveNode = LAMP_RED.mul(lamps.brake).mul(strip);
    // Reflections stronger than physical: without screen-space reflections the sky
    // reflection alone reads as flat black.
    glass.userData.envScale = 3;
    material.dispose();
    return glass;
  }
  if (/Glass|_Windows$/i.test(material.name)) {
    material.roughness = 0.02;
    material.metalness = 0;
  }
  return material;
}
