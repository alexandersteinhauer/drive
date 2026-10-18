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
    const floor = -car.cfg.cgHeight;
    const lower = box(1.62, 0.55, 4.3, paint);
    lower.position.set(0, floor + 0.45, 0.05);
    const cabin = box(1.36, 0.45, 2.1, 0x2b2f36);
    cabin.position.set(0, floor + 0.95, 0.35);
    const nose = box(1.5, 0.12, 1.2, paint);
    nose.position.set(0, floor + 0.78, -1.45);
    this.root.add(lower, cabin, nose);
    for (const x of [-0.65, 0.65]) {
      const head = box(0.35, 0.12, 0.05, 0xffffff, 0xfff2cc);
      head.position.set(x, floor + 0.55, -2.18);
      const tail = box(0.35, 0.1, 0.05, 0x550000, 0xaa0000);
      tail.position.set(x, floor + 0.6, 2.22);
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
