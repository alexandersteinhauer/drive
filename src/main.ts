import * as THREE from "three/webgpu";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { CarView } from "./carView";
import { Input } from "./input";
import { Car } from "./sim/car";
import { bmwI4, sportsCoupe } from "./sim/config";
import { RoadView } from "./roadView";
import { TerrainView } from "./terrainView";
import { Landscape } from "./world/landscape";

const DT = 1 / 480;
const SKY = 0xb4cde6;
/** How far ahead (along -Z) the road is generated beyond the car; must exceed the view radius. */
const ROAD_AHEAD = 1800;

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
document.body.prepend(renderer.domElement);
await renderer.init();
// Reflections for the car's paint, chrome and glass.
const carEnv = new THREE.PMREMGenerator(renderer).fromScene(
  new RoomEnvironment(),
  0.04,
).texture;

const scene = new THREE.Scene();
scene.background = new THREE.Color(SKY);
scene.fog = new THREE.Fog(SKY, 300, 1300);
const camera = new THREE.PerspectiveCamera(
  60,
  innerWidth / innerHeight,
  0.1,
  2500,
);

scene.add(new THREE.HemisphereLight(0xdfeaff, 0x4d5a3a, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, {
  left: -25,
  right: 25,
  top: 25,
  bottom: -25,
  near: 1,
  far: 150,
});
scene.add(sun, sun.target);

const land = new Landscape();
land.road.ensure(ROAD_AHEAD);
const terrain = new TerrainView(land);
const roadView = new RoadView(land);
scene.add(terrain.group, roadView.group);

const camDir = new THREE.Vector3(0, 0, -1);
const fwd = new THREE.Vector3();
const target = new THREE.Vector3();
const desired = new THREE.Vector3();
const orbit = new THREE.Vector3();
const camOffset = new THREE.Vector3(0, 2.2, 6.5);
const UP = new THREE.Vector3(0, 1, 0);
/** -1 left view, 0 behind, +1 right view (smoothed). */
let camSide = 0;

// C cycles through these.
const models = [
  { config: bmwI4, paint: 0x2f7fc4, model: "models/bmw-i4.glb" },
  { config: sportsCoupe, paint: 0xd2552d, model: undefined },
];
let model = 0;
let car = new Car(models[0]!.config, land);
placeOnRoad(15);
let view = makeView(0);
scene.add(view.root);
const input = new Input(window);
const speedEl = document.getElementById("speed")!;
if (import.meta.env.DEV)
  Object.assign(window, {
    car,
    land,
    placeOnRoad,
    terrain,
    roadView,
    renderer,
    camera,
    scene,
  });

/** Puts the car at rest in the right-hand lane at road sample `i`, facing along the road. */
function placeOnRoad(i: number): void {
  const road = land.road;
  while (road.finalCount <= i + 1) road.ensure(road.finalProgress + 200);
  const h = road.headings[i]!;
  const lane = road.cfg.halfWidth / 2;
  const at = new THREE.Vector3(
    road.xs[i]! + Math.cos(h) * lane,
    0,
    road.zs[i]! + Math.sin(h) * lane,
  );
  car.reset(at, -h);
}

function makeView(index: number): CarView {
  const { paint, model: url } = models[index]!;
  return new CarView(
    car,
    paint,
    url ? { url: import.meta.env.BASE_URL + url, envMap: carEnv } : undefined,
  );
}

function heading(): number {
  fwd.set(0, 0, -1).applyQuaternion(car.orientation);
  return Math.atan2(-fwd.x, -fwd.z);
}

function switchCar(): void {
  model = (model + 1) % models.length;
  const { config } = models[model]!;
  const next = new Car(config, land);
  next.reset(car.position, heading());
  scene.remove(view.root);
  car = next;
  view = makeView(model);
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
  if (input.take("KeyR"))
    placeOnRoad(land.road.nearestIndex(car.position.x, car.position.z));
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
  // Smooth the offset from the car, not the world position, so the camera doesn't fall
  // further behind the faster the car goes.
  desired.sub(pos);
  camOffset.lerp(desired, 1 - Math.exp(-10 * dt));
  camera.position.copy(pos).add(camOffset);
  // Keep the camera above the hills.
  const floorY = land.height(camera.position.x, camera.position.z) + 0.8;
  if (camera.position.y < floorY) camera.position.y = floorY;
  camera.lookAt(
    target
      .copy(pos)
      .addScaledVector(camDir, 2 * Math.cos(angle))
      .setY(pos.y + 0.6),
  );

  land.road.ensure(-pos.z + ROAD_AHEAD);
  terrain.update(pos, 4);
  roadView.update(pos);
  sun.position.set(pos.x + 40, pos.y + 70, pos.z + 25);
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

// Build the surroundings up front; after that, tiles stream in within a per-frame budget.
terrain.update(car.position, Infinity);
roadView.update(car.position);
renderer.setAnimationLoop(frame);
