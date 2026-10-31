import * as THREE from "three/webgpu";
import {
  convertToTexture,
  float,
  luminance,
  mix,
  pass,
  renderOutput,
  smoothstep,
  uniform,
  vec3,
  vec4,
} from "three/tsl";
import { bloom } from "three/addons/tsl/display/BloomNode.js";
import { gaussianBlur } from "three/addons/tsl/display/GaussianBlurNode.js";
import { smaa } from "three/addons/tsl/display/SMAANode.js";
import { toneMap } from "./tonemap";

/**
 * The crash look (see `createCrashPipeline`), set per frame. `flash` (0..1) floods the
 * whole picture with bright red, `light` dims it (1 full, 0 black), `blur` blends to the blurred image (0..1) and `spread` widens that blur. `black`
 * fades the normal picture to black (1), for when the car is back on the road.
 */
export const crash = {
  light: uniform(1),
  glow: uniform(1),
  blur: uniform(0),
  flash: uniform(0),
  spread: uniform(0),
  black: uniform(0),
};
/** Red of the highlights, and the (display-linear) luminance they start and peak at. */
const CRASH_RED = vec3(1, 0.04, 0.02);
const HIGHLIGHT_FROM = 0.12;
const HIGHLIGHT_FULL = 0.55;
/** Highlight brightness, above SDR white: saturated red, and brighter on an HDR display. */
const CRASH_BRIGHTNESS = 2.5;
/**
 * At full flash, the whole picture in pure red (any green or blue would wash it out to
 * white): its brightness gain, and a floor even in shadow.
 */
const FLASH_RED = vec3(1, 0, 0);
const FLASH_GAIN = 5;
const FLASH_FLOOR = 0.8;

/**
 * The picture after a fatal crash: black but for the highlights, in a bright, glowing red,
 * which `crash` then darkens and blurs. A pipeline of its own, so the blur costs nothing
 * while driving; `warmUpPipelines` (main.ts) builds its shaders at load.
 */
export function createCrashPipeline(
  renderer: THREE.WebGPURenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
): THREE.RenderPipeline {
  const color = pass(scene, camera).getTextureNode("output");
  const mapped = toneMap(color.rgb);
  const brightness = luminance(mapped);
  const red = CRASH_RED.mul(
    smoothstep(HIGHLIGHT_FROM, HIGHLIGHT_FULL, brightness).mul(
      CRASH_BRIGHTNESS,
    ),
  ).add(
    FLASH_RED.mul(brightness.mul(FLASH_GAIN).add(FLASH_FLOOR).mul(crash.flash)),
  );
  const image = convertToTexture(vec4(red, 1));
  const glow = bloom(image, crash.glow, 0.7, 0);
  const lit = image.rgb.add(glow.rgb);
  const blurred = gaussianBlur(vec4(lit, 1), crash.spread, 6, {
    resolutionScale: 0.5,
  });
  const pipeline = new THREE.RenderPipeline(renderer);
  pipeline.outputColorTransform = false;
  pipeline.outputNode = renderOutput(
    vec4(mix(lit, blurred.rgb, crash.blur).mul(crash.light), 1),
    THREE.NoToneMapping,
  );
  return pipeline;
}

/**
 * Scene render with a subtle bloom on bright highlights, tone mapping (tonemap.ts, with
 * HDR output), and SMAA anti-aliasing on the final image. No screen-space ambient occlusion: it cost ~3.5 ms of
 * a 8.3 ms (120 FPS) frame for a faint darkening of foliage and grass.
 */
export function createPipeline(
  renderer: THREE.WebGPURenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  /** For debugging: leave out stages. */
  skip: { bloom?: boolean; aa?: boolean } = {},
): THREE.RenderPipeline {
  const color = pass(scene, camera).getTextureNode("output");
  // Only real highlights (sun glints, sky) should glow, not sunlit white paint.
  const glow = bloom(color, 0.1, 0.3, 1.6);
  const pipeline = new THREE.RenderPipeline(renderer);
  pipeline.outputColorTransform = false;
  const lit = skip.bloom ? color : color.add(glow);
  const graded = renderOutput(
    vec4(toneMap(lit.rgb).mul(float(1).sub(crash.black)), 1),
    THREE.NoToneMapping,
  );
  pipeline.outputNode = skip.aa ? graded : smaa(graded);
  return pipeline;
}
