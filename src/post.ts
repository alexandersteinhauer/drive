import * as THREE from "three/webgpu";
import { pass, renderOutput, vec4 } from "three/tsl";
import { bloom } from "three/addons/tsl/display/BloomNode.js";
import { smaa } from "three/addons/tsl/display/SMAANode.js";
import { toneMap } from "./tonemap";

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
  const graded = renderOutput(vec4(toneMap(lit.rgb), 1), THREE.NoToneMapping);
  pipeline.outputNode = skip.aa ? graded : smaa(graded);
  return pipeline;
}
