import * as THREE from "three/webgpu";
import type { Car } from "./sim/car";

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

  constructor(
    private readonly car: Car,
    paint = 0xd2552d,
  ) {
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
    this.root.add(lower, cabin, nose);
    const front = cz - hz;
    const rear = cz + hz;
    for (const x of [-width * 0.38, width * 0.38]) {
      const head = box(width * 0.2, 0.1, 0.05, 0xffffff, 0xfff2cc);
      head.position.set(x, beltline - 0.12, front - 0.02);
      const tail = box(width * 0.22, 0.08, 0.05, 0x550000, 0xaa0000);
      tail.position.set(x, beltline - 0.06, rear + 0.02);
      this.root.add(head, tail);
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
      spin.add(tire, spoke, spoke2);
      steer.add(spin);
      steer.position.copy(w.mount);
      this.root.add(steer);
      this.wheels.push({ steer, spin });
    }
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
