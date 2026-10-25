import * as THREE from "three/webgpu";
import {
  float,
  neutralToneMapping,
  pow,
  toneMappingExposure,
  uniform,
  vec3,
} from "three/tsl";
import type { Node } from "three/webgpu";
import {
  ExtendedSRGBColorSpace,
  ExtendedSRGBColorSpaceImpl,
} from "three/addons/math/ColorSpaces.js";

/**
 * Whether the display can show HDR. If so, the canvas is a float16 extended-range surface
 * (extended sRGB): 1 is SDR white, and highlights go above it. Needs WebGPU; three's
 * WebGL fallback has no extended-range canvas.
 */
const hdrQuery = matchMedia("(dynamic-range: high)");
export const hdrCapable = "gpu" in navigator && hdrQuery.matches;

/**
 * How far above SDR white the brightest highlights go on an HDR display. Browsers don't
 * tell us the display's actual headroom (beyond it, the display clips), so this is a
 * modest value most HDR displays reach.
 */
const HDR_PEAK = 3;

/** Highlight ceiling, relative to SDR white: HDR_PEAK on an HDR display, else 1. */
const peak = uniform(1);

/** Renderer options for HDR output, when the display supports it. */
export function hdrRendererOptions(): {
  outputType?: typeof THREE.HalfFloatType;
} {
  return hdrCapable ? { outputType: THREE.HalfFloatType } : {};
}

/**
 * Switches the renderer to extended sRGB output if the display can show HDR, and follows
 * the window between HDR and SDR displays.
 */
export function setUpHdr(renderer: THREE.WebGPURenderer): void {
  const backend = renderer.backend as { isWebGPUBackend?: boolean };
  if (!hdrCapable || !backend.isWebGPUBackend) return;
  THREE.ColorManagement.define({
    [ExtendedSRGBColorSpace]: ExtendedSRGBColorSpaceImpl,
  });
  renderer.outputColorSpace = ExtendedSRGBColorSpace;
  const update = () => (peak.value = hdrQuery.matches ? HDR_PEAK : 1);
  hdrQuery.addEventListener("change", update);
  update();
}

/** The amount of the HDR highlight lift: none at SDR mid-tones, all of it at white. */
const HDR_ROLLOFF = 6;
const LUMA = vec3(0.2126, 0.7152, 0.0722);

/**
 * Tone mapping: Khronos PBR Neutral, which keeps colours true (AgX and ACES desaturate or
 * hue-shift this scene), from scene-linear to display-linear sRGB in 0 to 1. On an HDR
 * display the highlights are then stretched up to `peak`, scaling colours uniformly so
 * hues hold.
 */
export function toneMap(color: Node<"vec3">): Node<"vec3"> {
  const sdr = (
    neutralToneMapping(
      color,
      toneMappingExposure as unknown as Node<"float">,
    ) as Node<"vec3">
  ).toVar();
  const lift = pow(sdr.dot(LUMA), float(HDR_ROLLOFF));
  return sdr.mul(lift.mul(peak.sub(1)).add(1));
}
