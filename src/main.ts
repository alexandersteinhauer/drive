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
import { crash, createCrashPipeline, createPipeline } from "./post";
import { hdrRendererOptions, setUpHdr } from "./tonemap";
import { CarView, fetchModel } from "./carView";
import { Input } from "./input";
import { Car, type EscMode } from "./sim/car";
import {
  bmwI4,
  ferrari488Pista,
  vwBeetle,
  withImpulseExhaust,
  withPistaSurrogateExhaust,
  type CarConfig,
} from "./sim/config";
import { RoadView } from "./roadView";
import { TerrainView } from "./terrainView";
import { Vegetation } from "./vegetation";
import { Grass } from "./grass";
import { Landscape } from "./world/landscape";
import { CrashCard, Hud, type AssistMode } from "./hud";
import { PauseMenu } from "./pauseMenu";
import { Stats } from "./stats";
import { EngineSound } from "./audio/engineSound";
import { pistaExhaustMode, pistaExhaustUrl } from "./audio/exhaustOptions";
import { loadEngineModule } from "./sim/engineSim";
import type { EngineSpec } from "./sim/engineSpec";
import { SimulatedEngine } from "./sim/powerplant";
import { SunField } from "./sunField";
import { WorldWorkers } from "./worldWorkers";
import { LoadingScreen, filesLoaded } from "./loading";

const loading = new LoadingScreen();

/** ?bench: GPU benchmark instead of the game (see bench.ts). */
const bench = new URL(location.href).searchParams.has("bench")
  ? await import("./bench")
  : null;
bench?.instrument();

const DT = 1 / 480;
/**
 * After a fatal crash: the picture turns to glowing red highlights at once, which darken
 * and blur to black over CRASH_FADE (s). The black screen states what the crash did; after
 * CRASH_HOLD it waits for any key, then the car is back on the road and the picture fades
 * in over CRASH_RETURN.
 */
const CRASH_FADE = 2.5;
/** How fast the crash's red flash dies away (s, to 1/e). */
const CRASH_FLASH = 0.12;
/** How wide the crash blur gets (a multiple of its kernel). */
const CRASH_SPREAD = 4;
const CRASH_HOLD = 1;
const CRASH_RETURN = 0.6;
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
const crashPipeline = createCrashPipeline(renderer, scene, camera);
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
loading.enter("world", () => workers.pending);
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

/**
 * ?exhaust=impulse: the engines heard through engine-sim's exhaust impulse response
 * instead of the acoustics computed from each car's exhaust, to compare the two.
 */
const gameUrl = new URL(location.href);
const pistaExhaust = pistaExhaustMode(
  gameUrl,
  localStorage.getItem("pistaExhaust"),
);
const exhaust =
  gameUrl.searchParams.get("exhaust") === "impulse"
    ? withImpulseExhaust
    : (cfg: CarConfig) => cfg;
// C switches between these; the first is the one to start in.
const models: {
  config: CarConfig;
  url: string;
  headlights: { x: number; y: number; z: number };
}[] = [
  {
    config: bmwI4,
    url: "models/bmw-i4.glb",
    headlights: { x: 0.7, y: 0.72, z: -2.05 },
  },
  {
    config: exhaust(vwBeetle),
    url: "models/vw-beetle.glb",
    headlights: { x: 0.5, y: 0.78, z: -1.63 },
  },
  {
    config:
      pistaExhaust === "surrogate"
        ? withPistaSurrogateExhaust()
        : exhaust(ferrari488Pista),
    url: "models/ferrari-488-pista.glb",
    // At the cover, 0.16 m ahead of the projector lens: further in, the beam would light the
    // bumper below from inside.
    headlights: { x: 0.7, y: 0.57, z: -2.2 },
  },
];
// Shareable direct start, especially useful for comparing the optional Pista exhaust.
let model =
  gameUrl.searchParams.get("car") === "pista"
    ? 2
    : gameUrl.searchParams.get("car") === "beetle"
      ? 1
      : 0;
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
// Direct combustion starts need the module first; the default electric car warms it
// in the background, without delaying the first drive.
if (model !== 0) await loadEngineModule();
else
  void loadEngineModule().catch((error: unknown) =>
    console.warn("engine module unavailable:", error),
  );
let car = new Car(models[model]!.config, land);
applyAssists(car);
placeOnRoad(15);
let view = makeView(model);
scene.add(view.group);
const input = new Input(window);
const stats = new Stats(renderer);
const crashCard = new CrashCard();
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
    crashTime: () => crashTime,
    engineSound: () => engineSound,
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

function makeView(index: number, of = car): CarView {
  const { url, headlights } = models[index]!;
  return new CarView(of, {
    url: import.meta.env.BASE_URL + url,
    envMap: atmosphere.environment,
    headlights,
  });
}

function heading(): number {
  fwd.set(0, 0, -1).applyQuaternion(car.orientation);
  return Math.atan2(-fwd.x, -fwd.z);
}

/** The current car's engine sound, once loaded; null for a silent (electric) car. */
let engineSound: EngineSound | null = null;
/** A cached backend per engine spec; inactive cars are suspended, not left droning. */
const engineSounds = new Map<EngineSpec, Promise<EngineSound | null>>();
function simulatedEngine(of: Car): SimulatedEngine | null {
  return of.powerplant instanceof SimulatedEngine ? of.powerplant : null;
}
function loadEngineSound(): void {
  if (engineSound) {
    engineSound.onBackendChange = null;
    engineSound.onWarning = null;
    engineSound.stop();
  }
  engineSound = null;
  const engine = simulatedEngine(car);
  if (!engine) return;
  // Do not buffer an unconfirmed stream format while the optional backend initializes.
  engine.sim.collectAudio = false;
  const spec = engine.cfg.spec;
  let sound = engineSounds.get(spec);
  if (!sound) {
    sound = EngineSound.create(spec).catch((e: unknown) => {
      console.warn("engine sound unavailable:", e);
      if (spec.audio.surrogate)
        reportAudioWarning("Pista engine sound unavailable.");
      return null;
    });
    engineSounds.set(spec, sound);
  }
  const current = car;
  void sound.then((s) => {
    if (car !== current || !s) return;
    engineSound = s;
    if (spec.audio.surrogate) engine.sim.surrogateAudio = s.surrogate;
    engine.sim.collectAudio = true;
    s.onBackendChange = (surrogate) => {
      if (car !== current) return;
      engine.sim.surrogateAudio = surrogate;
      menu.setPistaExhaustStatus(
        surrogate
          ? "Exhaust-only pilot active; transients are experimental."
          : "Pilot unavailable; original exhaust active.",
      );
    };
    s.onWarning = reportAudioWarning;
    if (spec.audio.surrogate)
      menu.setPistaExhaustStatus(
        s.surrogate
          ? "Exhaust-only pilot active; transients are experimental."
          : "Pilot unavailable; original exhaust active.",
      );
    if (s.warning) reportAudioWarning(s.warning);
    if (audioInteracted) s.resume();
  });
}
// An interaction may happen before the worklet is ready; honor it once loaded.
let audioInteracted = false;
for (const type of ["keydown", "pointerdown"])
  addEventListener(type, () => {
    audioInteracted = true;
    engineSound?.resume();
  });
let audioNotice: HTMLElement | null = null;
function reportAudioWarning(message: string): void {
  menu.setPistaExhaustStatus(message);
  audioNotice?.remove();
  const notice = document.createElement("div");
  notice.setAttribute("role", "status");
  notice.textContent = message;
  notice.style.cssText =
    "position:fixed;bottom:90px;left:50%;transform:translateX(-50%);" +
    "max-width:85vw;padding:10px 16px;border-radius:8px;background:#18202ded;color:white;" +
    "font:13px/1.4 system-ui;pointer-events:none;z-index:10";
  audioNotice = notice;
  document.body.append(notice);
  setTimeout(() => notice.remove(), 10000);
}

/** Set while the next car's model loads; the current one drives on meanwhile. */
let switching = false;
async function switchCar(): Promise<void> {
  if (switching) return;
  switching = true;
  const index = (model + 1) % models.length;
  let next: Car;
  let nextView: CarView;
  try {
    await loadEngineModule();
    next = new Car(models[index]!.config, land);
    nextView = makeView(index, next);
    await nextView.ready;
    // Its shaders too, so the first frame with it doesn't stall.
    await renderer.compileAsync(nextView.group, camera, scene);
  } finally {
    switching = false;
  }
  model = index;
  applyAssists(next);
  next.reset(car.position, heading());
  scene.remove(view.group);
  const previous = car;
  car = next;
  previous.dispose();
  loadEngineSound();
  view = nextView;
  view.update(1);
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
  () => render(),
  {
    mode: pistaExhaust,
    status:
      pistaExhaust === "surrogate"
        ? "Exhaust-only pilot; known transient and training-rate limits."
        : "Original exhaust active.",
    onChange: (mode) => {
      localStorage.setItem("pistaExhaust", mode);
      location.assign(pistaExhaustUrl(new URL(location.href), mode));
    },
  },
);
loadEngineSound();
menu.enabled = false;
let last = performance.now();
let accumulator = 0;
/** Seconds since a fatal crash froze the car, or -1 when driving. */
let crashTime = -1;
/** Seconds since the car was put back on the road after a crash (fading back in). */
let returnTime = Infinity;

/** Advances the crash sequence by `dt` and sets the screen effect for it. */
function updateCrash(dt: number): void {
  if (crashTime >= 0) {
    // Once black, state what the crash was like.
    if (crashTime < CRASH_FADE && crashTime + dt >= CRASH_FADE)
      crashCard.show(car.impactSpeed);
    // Then wait for a key, pressed after the hold (not one still meant for driving).
    if (crashTime < WAIT_FROM && crashTime + dt >= WAIT_FROM) {
      input.takeAny();
      crashCard.prompt();
    }
    crashTime += dt;
  } else if (returnTime < CRASH_RETURN) returnTime += dt;
  if (crashTime >= 0) {
    // Darkening eases out (quickly at first, then lingering); the blur sets in over the first second
    // and keeps spreading; the glow dies down with the light.
    const t = Math.min(1, crashTime / CRASH_FADE);
    crash.light.value = (1 - t) ** 2;
    // A blinding red flash at the impact, gone within a few tenths of a second.
    crash.flash.value = Math.exp(-crashTime / CRASH_FLASH);
    crash.glow.value = 1.5 * (1 - t) + crash.flash.value;
    crash.blur.value = Math.min(1, t / 0.4);
    crash.spread.value = CRASH_SPREAD * Math.sqrt(t);
    crash.black.value = 1;
  } else {
    const t = Math.min(1, returnTime / CRASH_RETURN);
    crash.black.value = 1 - t * t * (3 - 2 * t);
  }
}

/** When the crash sequence starts waiting for a key (s). */
const WAIT_FROM = CRASH_FADE + CRASH_HOLD;

/** Ends the crash sequence: the car back on the road, the picture fading in. */
function respawn(): void {
  crashCard.hide();
  crashTime = -1;
  returnTime = 0;
  placeOnRoad(land.road.nearestIndex(car.position.x, car.position.z));
  hud.visible = true;
  accumulator = 0;
}

/** Draws the frame: the crash look while crashed, else the normal picture. */
function render(): void {
  // Fully black, waiting for a key: the canvas keeps the last (black) frame.
  if (crashTime > CRASH_FADE + 0.1) return;
  (crashTime >= 0 ? crashPipeline : pipeline).render();
}

function frame(now: number): void {
  if (menu.open) {
    engineSound?.setVolume(0);
    return;
  }
  stats.begin();
  const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
  last = now;
  const controls = input.update(dt);
  const crashed = crashTime >= 0;
  // After a crash, any key drives on (and does nothing else).
  if (crashTime >= WAIT_FROM && input.takeAny()) respawn();
  if (!crashed && input.take("KeyR"))
    placeOnRoad(land.road.nearestIndex(car.position.x, car.position.z));
  if (!crashed && input.take("KeyC")) void switchCar();
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
  // A fatal crash stops the physics at the step it happens in.
  if (paused || crashTime >= 0) accumulator = 0;
  let crashedNow = false;
  while (accumulator >= DT) {
    car.step(DT, controls);
    distance += Math.abs(car.speed) * DT;
    accumulator -= DT;
    if (car.fatal) {
      crashTime = 0;
      hud.visible = false;
      crashedNow = true;
      accumulator = 0;
      break;
    }
  }
  updateCrash(crashedNow || paused ? 0 : dt);
  const engine = simulatedEngine(car);
  if (engine) {
    // The engine's exhaust signal of this frame's physics steps.
    const samples = engine.sim.takeAudio();
    if (engineSound) {
      const stopped = crashTime >= 0 || paused;
      engineSound.setVolume(stopped ? 0 : 1);
      engineSound.push(samples);
    }
  }
  view.update(crashTime >= 0 ? 1 : accumulator / DT);

  const pos = view.root.position;
  // Chase camera: trails the car's heading with some lag. Holding Q/E swings it round to the left/right side.
  // Frozen with the car in a crash.
  if (crashTime < 0) {
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
  }

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

  render();
  stats.end();
}

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  menu.apply();
  // Paused, no frames are drawn: redraw the resized canvas once.
  if (menu.open) render();
});

// Build the surroundings up front (in the workers); after that they stream in a few
// pieces per frame. Wait for the models and textures too, then compile every shader
// before the first frame, so none is built mid-drive.
await Promise.all([
  terrain.prime(car.position),
  sunField.prime(car.position, atmosphere.lightDirection),
  vegetation.ready,
  view.ready,
  grass.prime(car.position),
]);
await filesLoaded();
vegetation.flush(car.position);
roadView.update(car.position);
loading.enter("shaders");
await renderer.compileAsync(scene, camera);
// Build the crash look's shaders now, not at the moment of a crash.
crashPipeline.render();
// Keys pressed while loading shouldn't act now; time starts from the first frame.
input.reset();
last = performance.now();
menu.enabled = true;
loading.finish();
// The other cars download in the background, ready for C.
for (const [i, { url }] of models.entries())
  if (i !== model) void fetchModel(import.meta.env.BASE_URL + url);
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
