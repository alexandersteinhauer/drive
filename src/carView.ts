import * as THREE from "three/webgpu";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import type { Car } from "./sim/car";

const CORNERS = ["fl", "fr", "rl", "rr"] as const;
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
  readonly root = new THREE.Group();
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
      for (const m of [o.material].flat())
        if (envMap && m instanceof THREE.MeshStandardMaterial)
          m.envMap = envMap;
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
    c.wheels.forEach((w, i) => {
      const v = this.wheels[i]!;
      v.steer.position.y = w.mount.y - w.length;
      v.steer.rotation.y = -w.steer;
      v.spin.rotation.x = -w.spin;
    });
  }
}
