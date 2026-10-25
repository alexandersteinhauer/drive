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
import { hdrRendererOptions, setUpHdr } from "./tonemap";
import { CarView } from "./carView";
import { Input } from "./input";
import { Car, type EscMode } from "./sim/car";
import { bmwI4, sportsCoupe } from "./sim/config";
import { RoadView } from "./roadView";
import { TerrainView } from "./terrainView";
import { Vegetation } from "./vegetation";
import { Grass } from "./grass";
import { Landscape } from "./world/landscape";
import { Hud, type AssistMode } from "./hud";
import { PauseMenu } from "./pauseMenu";
import { Stats } from "./stats";
import { SunField } from "./sunField";
import { WorldWorkers } from "./worldWorkers";

/** ?bench: GPU benchmark instead of the game (see bench.ts). */
const bench = new URL(location.href).searchParams.has("bench")
  ? await import("./bench")
  : null;
bench?.instrument();

const DT = 1 / 480;
/** How far ahead (along -Z) the road is generated beyond the car; must exceed the view radius. */
const ROAD_AHEAD = 2600;

// Anti-aliasing happens in post-processing (SMAA), so the canvas itself needs no MSAA.
const renderer = new THREE.WebGPURenderer({
  antialias: false,
  ...hdrRendererOptions(),
});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
// Tone mapping (Khronos PBR Neutral) is done in post.ts; this exposure feeds it.
const EXPOSURE = 0.65;
/** Hours of game time per second while [ or ] is held. */
const TIME_RATE = 2;
renderer.toneMappingExposure = EXPOSURE;
document.body.prepend(renderer.domElement);
await renderer.init();
setUpHdr(renderer);

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

/**
 * The world seed comes from the URL (?seed=123), so a world can be shared or revisited;
 * without one a random seed is picked and written into the URL.
 */
function worldSeed(): number {
  const url = new URL(location.href);
  const given = Number(url.searchParams.get("seed"));
  if (url.searchParams.has("seed") && Number.isSafeInteger(given)) return given;
  const seed = 1 + Math.floor(Math.random() * 999_999);
  url.searchParams.set("seed", String(seed));
  history.replaceState(history.state, "", url);
  return seed;
}
const land = new Landscape(worldSeed());
land.road.ensure(ROAD_AHEAD);
// Terrain, vegetation, grass and the sun field's tiles are generated off the main thread.
const workers = new WorldWorkers(land);
const sunField = new SunField(renderer, workers);
atmosphere.useField(sunField);
const vegetation = new Vegetation(
  import.meta.env.BASE_URL + "models/nature.glb",
  renderer,
);
const terrain = new TerrainView(land, vegetation, workers);
const roadView = new RoadView(land);
const grass = new Grass(workers);
scene.add(terrain.group, roadView.group, vegetation.group, grass.group);

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
// Driver assists, kept across car switches and reloads.
const assists: Record<"abs" | "esc", AssistMode> = {
  abs: localStorage.getItem("abs") === "off" ? "off" : "on",
  esc:
    (["on", "sport", "off"] as const).find(
      (m) => m === localStorage.getItem("esc"),
    ) ?? "on",
};
function applyAssists(target: Car): void {
  target.absEnabled = assists.abs !== "off";
  target.escMode = assists.esc as EscMode;
}
let car = new Car(models[0]!.config, land);
applyAssists(car);
placeOnRoad(15);
let view = makeView(0);
scene.add(view.group);
const input = new Input(window);
const stats = new Stats(renderer);
const hud = new Hud(land.road, assists, (assist, mode) => {
  applyAssists(car);
  localStorage.setItem(assist, mode);
});
if (import.meta.env.DEV)
  Object.assign(window, {
    car,
    land,
    placeOnRoad,
    terrain,
    vegetation,
    grass,
    roadView,
    sunField,
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
  applyAssists(next);
  next.reset(car.position, heading());
  scene.remove(view.group);
  car = next;
  view = makeView(model);
  scene.add(view.group);
  if (import.meta.env.DEV) Object.assign(window, { car });
}

/** Dev hook: freeze physics (for reproducible captures). */
let paused = false;
/** Distance driven (m), for the HUD's odometer; teleports back onto the road don't count.
 * Kept across visits: saved about once a second while driving, and when the page is left. */
let distance = Number(localStorage.getItem("distance")) || 0;
let savedDistance = distance;
let distanceSavedAt = 0;
/** Adds what was driven since the last save to the stored total, so tabs open at once
 * don't overwrite each other's distance. */
function saveDistance(): void {
  if (distance === savedDistance) return;
  const stored = Number(localStorage.getItem("distance")) || 0;
  localStorage.setItem(
    "distance",
    (stored + distance - savedDistance).toFixed(1),
  );
  savedDistance = distance;
}
addEventListener("pagehide", saveDistance);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) saveDistance();
});
/** Escape opens the pause menu: the last frame stays up, blurred, behind it. */
const menu = new PauseMenu(
  renderer,
  () => {
    // Keys pressed while paused shouldn't act on resume; time restarts from now.
    input.reset();
    last = performance.now();
  },
  // Paused, no frames are drawn: redraw once to show a new setting.
  () => pipeline.render(),
);
let last = performance.now();
let accumulator = 0;

function frame(now: number): void {
  if (menu.open) return;
  stats.begin();
  const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
  last = now;
  const controls = input.update(dt);
  if (input.take("KeyR"))
    placeOnRoad(land.road.nearestIndex(car.position.x, car.position.z));
  if (input.take("KeyC")) switchCar();
  if (input.take("KeyF")) stats.visible = !stats.visible;
  if (input.take("KeyM")) hud.minimap.visible = !hud.minimap.visible;
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
    distance += Math.abs(car.speed) * DT;
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
  terrain.update(pos);
  vegetation.update(pos);
  roadView.update(pos);
  atmosphere.update(camera);
  atmosphere.follow(view.root.position);
  sunField.update(camera.position, atmosphere.lightDirection);
  vegetation.setSun(atmosphere.sun);
  vegetation.setViewer(camera.position);
  grass.setSun(atmosphere.sun);
  grass.update(camera.position);
  // The car's sky light/reflections: strong in sunlight (sparkly paint), but down to the
  // terrain's level in a hill's shadow, where it otherwise glows against its surroundings.
  const sunlit = atmosphere.sunVisibility(car.position, (x, z) =>
    land.height(x, z),
  );
  carSun += (sunlit - carSun) * (1 - Math.exp(-4 * dt));
  view.setEnvironmentIntensity(
    THREE.MathUtils.lerp(scene.environmentIntensity, CAR_REFLECTIONS, carSun),
  );

  hud.minimap.update(pos.x, pos.z, camDir.x, camDir.z, car.speed, dt);
  hud.setSpeed(car.speed);
  hud.setDistance(distance);
  if (now - distanceSavedAt > 1000) {
    distanceSavedAt = now;
    saveDistance();
  }

  pipeline.render();
  stats.end();
}

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  menu.apply();
  // Paused, no frames are drawn: redraw the resized canvas once.
  if (menu.open) pipeline.render();
});

// Build the surroundings up front (in the workers); after that they stream in a few
// pieces per frame. Then compile every shader before the first frame, so none is built
// mid-drive.
await Promise.all([
  terrain.prime(car.position),
  sunField.prime(car.position, atmosphere.lightDirection),
  vegetation.ready,
  grass.prime(car.position),
]);
vegetation.flush(car.position);
roadView.update(car.position);
await renderer.compileAsync(scene, camera);
if (bench) {
  paused = true;
  stats.visible = false;
  const result = await bench.run(
    {
      renderer,
      scene,
      frame,
      goTo: async (i) => {
        placeOnRoad(i);
        land.road.ensure(-car.position.z + ROAD_AHEAD);
        view.update(1);
        await Promise.all([
          terrain.prime(car.position),
          sunField.prime(car.position, atmosphere.lightDirection),
          grass.prime(car.position),
        ]);
        vegetation.flush(car.position);
        roadView.update(car.position);
        await renderer.compileAsync(scene, camera);
      },
      setPost,
      setTime: (hours) => atmosphere.setTime(hours),
      compile: () => renderer.compileAsync(scene, camera),
      toggles: {},
      parts: {
        terrain: [terrain.group],
        road: [roadView.group],
        "near trees": vegetation.group.children.filter(
          (o) => (o as THREE.InstancedMesh).isInstancedMesh,
        ),
        "far trees": vegetation.group.children.filter(
          (o) => !(o as THREE.InstancedMesh).isInstancedMesh,
        ),
        grass: [grass.group],
        car: [view.group],
        sky: [atmosphere.sky],
      },
    },
    new URL(location.href).searchParams.get("bench") ?? "",
  );
  console.log(result);
  Object.assign(window, { benchResult: result });
} else renderer.setAnimationLoop(frame);
