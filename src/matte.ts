import * as THREE from "three/webgpu";
import {
  BRDF_Lambert,
  diffuseColor,
  float,
  normalView,
  positionViewDirection,
  pow,
  vec3,
} from "three/tsl";
import type { NodeBuilder } from "three/webgpu";

/** A vec3 node (the lighting context's accumulators are loosely typed). */
type V3 = ReturnType<typeof vec3>;

const lambert = () =>
  BRDF_Lambert({ diffuseColor: diffuseColor.rgb }) as unknown as V3;

/**
 * The share of light left for the diffuse layer once a dielectric's specular reflection
 * (4% head-on, all of it at grazing angles: Schlick) has taken its part, as the physical
 * model has it. Without it, matte surfaces seen at a slant come out brighter.
 */
/** Sky reflection added at grazing angles (see MatteNodeMaterial.sheen). */
const SKY_SHEEN_GRAZING = 0.08;
const diffuseShare = (cos: V3) =>
  float(1).sub(
    float(0.04).add(float(0.96).mul(pow(float(1).sub(cos).max(0), 5))),
  ) as unknown as V3;

/**
 * Lambert-only physically based lighting: the sun (with its shadows), the fill light and
 * the sky's diffuse light, as MeshStandardNodeMaterial gives them, but without the specular
 * lobes, their environment reflection lookup and the BRDF table. For matte, alpha-tested
 * foliage (roughness ~0.9), where those add next to nothing on screen but cost per pixel.
 */
class MatteLightingModel extends THREE.LightingModel {
  constructor(private readonly sheen: number) {
    super();
  }

  override direct(light: THREE.LightingModelDirectInput): void {
    const l = light.lightDirection as unknown as V3;
    const dotNL = normalView.dot(l).clamp() as unknown as V3;
    const half = l.add(positionViewDirection).normalize();
    const dotVH = positionViewDirection.dot(half).clamp() as unknown as V3;
    (light.reflectedLight.directDiffuse as unknown as V3).addAssign(
      dotNL
        .mul(light.lightColor as unknown as V3)
        .mul(lambert())
        .mul(diffuseShare(dotVH)),
    );
  }

  override indirect(builder: NodeBuilder): void {
    const c = builder.context as unknown as Record<
      "irradiance" | "iblIrradiance" | "ambientOcclusion",
      V3
    > & { reflectedLight: Record<"indirectDiffuse" | "indirectSpecular", V3> };
    const dotNV = normalView.dot(positionViewDirection).abs() as unknown as V3;
    // Hemisphere/ambient lights and the sky's PMREM irradiance, both diffuse, less what
    // the surface reflects. Rough surfaces reflect little even at grazing angles (the
    // roughness-aware Schlick: 0.04 head-on, up to ~0.15), plus some multiple scattering,
    // and what they reflect is the sky blurred to its irradiance: no radiance lookup.
    const f = float(this.sheen).add(
      float(SKY_SHEEN_GRAZING).mul(pow(float(1).sub(dotNV).max(0), 5)),
    ) as unknown as V3;
    const share = float(1).sub(f) as unknown as V3;
    c.reflectedLight.indirectDiffuse.addAssign(
      c.irradiance.add(c.iblIrradiance).mul(lambert()).mul(share),
    );
    c.reflectedLight.indirectSpecular.addAssign(
      c.iblIrradiance.mul(1 / Math.PI).mul(f),
    );
    c.reflectedLight.indirectDiffuse.mulAssign(c.ambientOcclusion);
    c.reflectedLight.indirectSpecular.mulAssign(c.ambientOcclusion);
  }
}

/** MeshStandardNodeMaterial lit diffuse-only (see MatteLightingModel). */
export class MatteNodeMaterial extends THREE.MeshStandardNodeMaterial {
  /**
   * Sky reflection head-on: 0.07 matches MeshStandardNodeMaterial at roughness ~0.9 on
   * leaves (by eye and average colour); grass, lit as the ground under it and seen edge-on,
   * takes less or the meadow turns blue.
   */
  sheen = 0.07;

  override setupLightingModel(): THREE.PhysicalLightingModel {
    return new MatteLightingModel(
      this.sheen,
    ) as unknown as THREE.PhysicalLightingModel;
  }
}
