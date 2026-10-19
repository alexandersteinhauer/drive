import * as THREE from "three/webgpu";
import {
  unpackRGBToNormal,
  packNormalToRGB,
  mrt,
  normalView,
  output,
  pass,
  renderOutput,
  sample,
  vec4,
} from "three/tsl";
import { ao } from "three/addons/tsl/display/GTAONode.js";
import { bloom } from "three/addons/tsl/display/BloomNode.js";
import { denoise } from "three/addons/tsl/display/DenoiseNode.js";
import { smaa } from "three/addons/tsl/display/SMAANode.js";

/**
 * Scene render with ambient occlusion (half resolution), a subtle bloom on bright highlights,
 * tone mapping, and SMAA anti-aliasing on the final image.
 */
export function createPipeline(
  renderer: THREE.WebGPURenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  /** For debugging: leave out stages. */
  skip: { ao?: boolean; bloom?: boolean; aa?: boolean } = {},
): THREE.RenderPipeline {
  const scenePass = pass(scene, camera);
  scenePass.setMRT(mrt({ output, normal: packNormalToRGB(normalView) }));
  const color = scenePass.getTextureNode("output");
  const depth = scenePass.getTextureNode("depth");
  const normals = scenePass.getTextureNode("normal");

  const occlusion = ao(
    depth,
    sample((uv) => unpackRGBToNormal(normals.sample(uv))),
    camera,
  );
  occlusion.resolutionScale = 0.5;
  occlusion.radius.value = 0.8;
  occlusion.thickness.value = 1.5;
  occlusion.distanceFallOff.value = 0.5;
  const normal = sample((uv) => unpackRGBToNormal(normals.sample(uv)));
  // Half-resolution AO is speckled; a depth/normal-aware blur smooths it without bleeding.
  const smoothAo = denoise(occlusion.getTextureNode(), depth, normal, camera);
  // DenoiseNode outputs a vec4 but its typings don't say so.
  // Clamped: the blur can overshoot above 1 at silhouettes against the sky, which brightens
  // edges and then blooms into a halo.
  const aoValue = (smoothAo as unknown as ReturnType<typeof vec4>).r.clamp(
    0,
    1,
  );
  const occluded = skip.ao ? color : vec4(color.rgb.mul(aoValue), color.a);

  // Only real highlights (sun glints, sky) should glow, not sunlit white paint.
  const glow = bloom(occluded, 0.1, 0.3, 1.6);
  const pipeline = new THREE.RenderPipeline(renderer);
  pipeline.outputColorTransform = false;
  const graded = renderOutput(skip.bloom ? occluded : occluded.add(glow));
  pipeline.outputNode = skip.aa ? graded : smaa(graded);
  return pipeline;
}
