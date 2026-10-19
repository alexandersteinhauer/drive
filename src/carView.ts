import * as THREE from "three/webgpu";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import type { Car } from "./sim/car";

const CORNERS = ["fl", "fr", "rl", "rr"] as const;
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
  /**
   * A soft dark patch on the ground under the car. Shadow maps are too coarse for the
   * contact shading right under the sills and tyres; without it the car looks like it floats.
   */
  private readonly contact: THREE.Mesh;
  private readonly wheels: { steer: THREE.Group; spin: THREE.Group }[] = [];
  /** Boxes shown until (or instead of) a model. */
  private readonly placeholder = new THREE.Group();

  constructor(
    private readonly car: Car,
    paint = 0xd2552d,
    /** Optional GLB with nodes "body", "wheel_<fl|fr|rl|rr>_spin" and "wheel_<..>_caliper". */
    model?: { url: string; envMap?: THREE.Texture },
  ) {
    this.root.add(this.placeholder);
    this.contact = contactShadow(car);
    this.group.add(this.root, this.contact);
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
    if (model) void this.load(model.url, model.envMap);
  }

  /**
   * Swaps the placeholder for a model. Its origin is on the ground midway between the axles;
   * wheel parts are centred on their wheel centres and scaled to the physics wheel radius.
   */
  private async load(url: string, envMap?: THREE.Texture): Promise<void> {
    const gltf = await loader.loadAsync(url);
    const scene = gltf.scene;
    const info = scene.userData as {
      radius: number;
      wheels: Record<(typeof CORNERS)[number], { center: number[] }>;
    };
    scene.traverse((o) => {
      if (!(o instanceof THREE.Mesh)) return;
      o.castShadow = true;
      o.receiveShadow = true;
      o.material = upgrade(o.material as THREE.Material, envMap);
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
  }

  /** `alpha` blends between the previous and current physics state. */
  update(alpha: number): void {
    const c = this.car;
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
 * surface; the source materials are plain and read as matte plastic.
 */
function upgrade(
  material: THREE.Material,
  envMap?: THREE.Texture,
): THREE.Material {
  if (!(material instanceof THREE.MeshStandardMaterial)) return material;
  if (envMap) material.envMap = envMap;
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
  if (/Glass|Windows/i.test(material.name)) {
    material.roughness = 0.02;
    material.metalness = 0;
  }
  return material;
}
