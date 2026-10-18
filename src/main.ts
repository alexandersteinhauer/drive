import * as THREE from "three/webgpu";
import { CarView } from "./carView";
import { Input } from "./input";
import { Car } from "./sim/car";
import { electricCoupe, sportsCoupe } from "./sim/config";
import { FlatGround } from "./sim/ground";

const DT = 1 / 480;
const SKY = 0xbfd4e6;

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(SKY);
scene.fog = new THREE.Fog(SKY, 80, 190);
const camera = new THREE.PerspectiveCamera(
  60,
  innerWidth / innerHeight,
  0.1,
  500,
);

scene.add(new THREE.HemisphereLight(0xdfeaff, 0x5a5348, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, {
  left: -15,
  right: 15,
  top: 15,
  bottom: -15,
  near: 1,
  far: 80,
});
scene.add(sun, sun.target);

// Ground and grids; they follow the car, snapped to the major grid, so the plane is endless.
const floor = new THREE.Group();
const plane = new THREE.Mesh(
  new THREE.PlaneGeometry(500, 500),
  new THREE.MeshStandardMaterial({ color: 0x8a8f86, roughness: 1 }),
);
plane.rotation.x = -Math.PI / 2;
plane.receiveShadow = true;
const minor = new THREE.GridHelper(400, 400, 0x7a7f77, 0x7a7f77);
const major = new THREE.GridHelper(400, 40, 0x50544f, 0x50544f);
minor.position.y = 0.005;
major.position.y = 0.01;
floor.add(plane, minor, major);
scene.add(floor);

const camDir = new THREE.Vector3(0, 0, -1);
const fwd = new THREE.Vector3();
const target = new THREE.Vector3();
const desired = new THREE.Vector3();
const orbit = new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);
/** -1 left view, 0 behind, +1 right view (smoothed). */
let camSide = 0;

// C cycles through these.
const models = [
  { config: sportsCoupe, paint: 0xd2552d },
  { config: electricCoupe, paint: 0x2f7fc4 },
];
let model = 0;
const ground = new FlatGround();
let car = new Car(sportsCoupe, ground);
let view = new CarView(car, models[0]!.paint);
scene.add(view.root);
const input = new Input(window);
const speedEl = document.getElementById("speed")!;
if (import.meta.env.DEV) Object.assign(window, { car });

function heading(): number {
  fwd.set(0, 0, -1).applyQuaternion(car.orientation);
  return Math.atan2(-fwd.x, -fwd.z);
}

function switchCar(): void {
  model = (model + 1) % models.length;
  const { config, paint } = models[model]!;
  const next = new Car(config, ground);
  next.reset(car.position, heading());
  scene.remove(view.root);
  car = next;
  view = new CarView(car, paint);
  scene.add(view.root);
  if (import.meta.env.DEV) Object.assign(window, { car });
}

let last = performance.now();
let accumulator = 0;
let shownSpeed = -1;

function frame(now: number): void {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  const controls = input.update(dt);
  if (input.take("KeyR")) car.reset(car.position, heading());
  if (input.take("KeyC")) switchCar();
  accumulator += dt;
  while (accumulator >= DT) {
    car.step(DT, controls);
    accumulator -= DT;
  }
  view.update(accumulator / DT);

  // Chase camera: trails the car's heading with some lag. Holding Q/E swings it round to the left/right side.
  const pos = view.root.position;
  fwd.set(0, 0, -1).applyQuaternion(view.root.quaternion).setY(0).normalize();
  camDir.lerp(fwd, 1 - Math.exp(-4 * dt)).normalize();
  const sideTarget =
    (input.held("KeyE") ? 1 : 0) - (input.held("KeyQ") ? 1 : 0);
  camSide += (sideTarget - camSide) * (1 - Math.exp(-8 * dt));
  const angle = (camSide * Math.PI) / 2;
  orbit.copy(camDir).negate().applyAxisAngle(UP, angle);
  desired
    .copy(pos)
    .addScaledVector(orbit, 6.5)
    .setY(pos.y + 2.2 - 0.8 * Math.abs(camSide));
  camera.position.lerp(desired, 1 - Math.exp(-10 * dt));
  camera.lookAt(
    target
      .copy(pos)
      .addScaledVector(camDir, 2 * Math.cos(angle))
      .setY(pos.y + 0.6),
  );

  floor.position.set(
    Math.round(pos.x / 10) * 10,
    0,
    Math.round(pos.z / 10) * 10,
  );
  sun.position.set(pos.x + 20, 40, pos.z + 10);
  sun.target.position.copy(pos);

  const kmh = Math.round(Math.abs(car.speed) * 3.6);
  if (kmh !== shownSpeed) speedEl.textContent = String((shownSpeed = kmh));

  renderer.render(scene, camera);
}

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

renderer.setAnimationLoop(frame);
