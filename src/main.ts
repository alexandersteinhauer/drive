import * as THREE from "three/webgpu";

// Dev: keep console errors/warnings around so they can be read back from tooling.
if (import.meta.env.DEV) {
  const log: string[] = [];
  Object.assign(window, { __errors: log });
  for (const level of ["error", "warn"] as const) {
    const original = console[level].bind(console);
    console[level] = (...args: unknown[]) => {
      log.push(`${level}: ${args.map(String).join(" ").slice(0, 500)}`);
      original(...args);
    };
  }
}
import { Atmosphere, skyGain, skySaturation } from "./atmosphere";
import { createPipeline } from "./post";
import { CarView } from "./carView";
import { Input } from "./input";
import { Car } from "./sim/car";
import { bmwI4, sportsCoupe } from "./sim/config";
import { RoadView } from "./roadView";
import { TerrainView } from "./terrainView";
import { Vegetation } from "./vegetation";
import { Landscape } from "./world/landscape";

const DT = 1 / 480;
/** How far ahead (along -Z) the road is generated beyond the car; must exceed the view radius. */
const ROAD_AHEAD = 2600;

// Anti-aliasing happens in post-processing (SMAA), so the canvas itself needs no MSAA.
const renderer = new THREE.WebGPURenderer({ antialias: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
// Khronos PBR Neutral keeps colours true (AgX/ACES desaturate this scene into a milky look).
renderer.toneMapping = THREE.NeutralToneMapping;
const EXPOSURE = 0.65;
/** Hours of game time per second while [ or ] is held. */
const TIME_RATE = 2;
renderer.toneMappingExposure = EXPOSURE;
document.body.prepend(renderer.domElement);
await renderer.init();

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  60,
  innerWidth / innerHeight,
  0.1,
  3200,
);
const atmosphere = new Atmosphere(renderer, scene, {
  time: Number(localStorage.getItem("time") ?? 17) || 17,
  // Roughly a 2-2.5x lit/shadow contrast after tone mapping, as on a clear afternoon.
  sunLight: 7,
  skyLight: 0.22,
  fillLight: 0.9,
  hazeHalfDistance: 8000,
  farDistance: 2150,
});
let pipeline = createPipeline(renderer, scene, camera);
/** Dev hook: rebuild the post-processing chain without some stages. */
function setPost(skip: Parameters<typeof createPipeline>[3]): void {
  pipeline = createPipeline(renderer, scene, camera, skip);
}

const land = new Landscape();
land.road.ensure(ROAD_AHEAD);
const vegetation = new Vegetation(
  land,
  import.meta.env.BASE_URL + "models/nature.glb",
  renderer,
);
const terrain = new TerrainView(land, vegetation);
const roadView = new RoadView(land);
scene.add(terrain.group, roadView.group, vegetation.group);

const camDir = new THREE.Vector3(0, 0, -1);
const fwd = new THREE.Vector3();
const target = new THREE.Vector3();
const desired = new THREE.Vector3();
const orbit = new THREE.Vector3();
const camOffset = new THREE.Vector3(0, 2.2, 6.5);
const UP = new THREE.Vector3(0, 1, 0);
/** Car sky-light/reflection strength in full sun. */
const CAR_REFLECTIONS = 0.7;
/** How much sun reaches the car (smoothed). */
let carSun = 1;
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
scene.add(view.group);
const input = new Input(window);
const speedEl = document.getElementById("speed")!;
if (import.meta.env.DEV)
  Object.assign(window, {
    car,
    land,
    placeOnRoad,
    terrain,
    vegetation,
    roadView,
    renderer,
    camera,
    scene,
    get pipeline() {
      return pipeline;
    },
    setPost,
    pause: (on: boolean) => (paused = on),
    atmosphere,
    skySaturation,
    skyGain,
    setTime: (hours: number) => atmosphere.setTime(hours),
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
    url
      ? { url: import.meta.env.BASE_URL + url, envMap: atmosphere.environment }
      : undefined,
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
  scene.remove(view.group);
  car = next;
  view = makeView(model);
  scene.add(view.group);
  if (import.meta.env.DEV) Object.assign(window, { car });
}

/** Dev hook: freeze physics (for reproducible captures). */
let paused = false;
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
  // Hold [ / ] to turn time back / forward, two hours per second.
  const timeStep =
    (input.held("BracketRight") ? 1 : 0) - (input.held("BracketLeft") ? 1 : 0);
  if (timeStep) {
    atmosphere.setTime(atmosphere.time + timeStep * TIME_RATE * dt);
    localStorage.setItem("time", atmosphere.time.toFixed(2));
  }
  renderer.toneMappingExposure = EXPOSURE * atmosphere.exposure;
  view.setHeadlights(atmosphere.darkness);
  accumulator += dt;
  if (paused) accumulator = 0;
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
  terrain.update(pos, 2.5);
  vegetation.update(pos);
  roadView.update(pos);
  atmosphere.update(camera);
  vegetation.setSun(atmosphere.sun);
  // The car's sky light/reflections: strong in sunlight (sparkly paint), but down to the
  // terrain's level in a hill's shadow, where it otherwise glows against its surroundings.
  const sunlit = atmosphere.sunVisibility(car.position, (x, z) =>
    land.height(x, z),
  );
  carSun += (sunlit - carSun) * (1 - Math.exp(-4 * dt));
  view.setEnvironmentIntensity(
    THREE.MathUtils.lerp(scene.environmentIntensity, CAR_REFLECTIONS, carSun),
  );

  const kmh = Math.round(Math.abs(car.speed) * 3.6);
  if (kmh !== shownSpeed) speedEl.textContent = String((shownSpeed = kmh));

  pipeline.render();
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
