import * as THREE from "three/webgpu";
import {
  cameraPosition,
  cubeTexture,
  exp,
  float,
  fog,
  max,
  positionWorld,
  mix,
  smoothstep,
  uniform,
  vec3,
  vec4,
} from "three/tsl";
import type { Node, UniformNode } from "three/webgpu";
import { CSMShadowNode } from "three/addons/csm/CSMShadowNode.js";
import { SkyMesh } from "three/addons/objects/SkyMesh.js";

export interface AtmosphereOptions {
  /** Local solar time in hours (0-24). */
  time: number;
  /** Haze: half the light is scattered away after this many metres. */
  hazeHalfDistance: number;
  /** Everything is fully hazed out by this distance (hides the terrain edge). */
  farDistance: number;
  /** Sun intensity at high elevation, and the strength of the sky's image-based light. */
  sunLight: number;
  skyLight: number;
  fillLight: number;
}

/** The scene's latitude and the sun's declination (early summer), in degrees. */
const LATITUDE = 47;
const DECLINATION = 18;
/** The road's main direction (-Z) is this many degrees west of north. */
const ROAD_BEARING = 31;

/**
 * Sun elevation and azimuth (degrees) at a local solar time, as seen from the scene: azimuth
 * 0 is behind the road's main direction (+Z), 90 is to the right (+X).
 */
export function sunAngles(hours: number): {
  elevation: number;
  azimuth: number;
} {
  const rad = THREE.MathUtils.degToRad;
  const lat = rad(LATITUDE);
  const dec = rad(DECLINATION);
  const h = rad((hours - 12) * 15);
  const elevation = Math.asin(
    Math.sin(lat) * Math.sin(dec) + Math.cos(lat) * Math.cos(dec) * Math.cos(h),
  );
  const azimuth = Math.atan2(
    Math.sin(h),
    Math.cos(h) * Math.sin(lat) - Math.tan(dec) * Math.cos(lat),
  );
  return {
    elevation: THREE.MathUtils.radToDeg(elevation),
    azimuth: THREE.MathUtils.radToDeg(azimuth) - ROAD_BEARING,
  };
}

/**
 * Physically based sky, a sun that matches it, image-based lighting from the same sky,
 * and haze that fades into the sky colour in each view direction.
 */
export class Atmosphere {
  readonly sky = new SkyMesh();
  readonly sun = new THREE.DirectionalLight(0xffffff, 3);
  readonly sunDirection = new THREE.Vector3();
  /** Pre-filtered sky for image-based lighting. */
  readonly environment: THREE.Texture;
  private readonly cube: THREE.CubeRenderTarget;

  /** How dark it is: 0 in daylight, 1 at night (for exposure and headlights). */
  darkness = 0;
  /** Exposure multiplier that adapts to the light, like the eye. */
  exposure = 1;
  private readonly renderer: THREE.WebGPURenderer;
  private readonly options: AtmosphereOptions;
  private readonly fill: THREE.HemisphereLight;
  private readonly skyCopy = new SkyMesh();
  private readonly capture = new THREE.Scene();
  private readonly cubeCamera: THREE.CubeCamera;
  private readonly pmrem: THREE.PMREMGenerator;
  private readonly envTarget: THREE.RenderTarget;
  private captured = NaN;
  /** 0 by day, 1 at night: the sky model turns brown below the horizon, so it is tinted. */
  private readonly night = uniform(0);
  private readonly twilight = uniform(0);
  /** Haze extinction per metre. */
  readonly hazeDensity = uniform(0);
  /** How far above the horizon (direction y) the haze takes its colour from. */
  readonly hazeLift = uniform(0.08);
  private hours = NaN;

  constructor(
    renderer: THREE.WebGPURenderer,
    scene: THREE.Scene,
    options: AtmosphereOptions,
  ) {
    this.renderer = renderer;
    this.options = options;
    this.hazeDensity.value = Math.LN2 / options.hazeHalfDistance;
    configureSky(this.sky, this.night, this.twilight);
    this.sky.scale.setScalar(2000);
    scene.add(this.sky);

    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0003;
    this.sun.shadow.normalBias = 0.05;
    // Cascaded shadows: sharp near the car, still present on hills hundreds of metres away.
    // Split by hand so the first cascade covers only the car and its surroundings (~2 cm
    // texels); the automatic split gave it ~75 m and a blurry car shadow. Three cascades,
    // since each costs the terrain shader a texture.
    const csm = new CSMShadowNode(this.sun, {
      cascades: 3,
      maxFar: 450,
      mode: "custom",
      customSplitsCallback: (
        _cascades: number,
        _near: number,
        far: number,
        breaks: number[],
      ) => breaks.push(...SHADOW_SPLITS.map((d) => Math.min(1, d / far))),
      lightMargin: 150,
    });
    csm.fade = true;
    this.sun.shadow.shadowNode = csm;
    scene.add(this.sun, this.sun.target);
    // Soft neutral fill (sky above, warm earth bounce below): lifts shaded slopes out of a
    // murky blue-teal without brightening reflections the way more sky light would.
    this.fill = new THREE.HemisphereLight(
      0xece9e0,
      0x8a7a5c,
      options.fillLight,
    );
    scene.add(this.fill);

    // The sky without its sun disc, captured into a cube map whenever the time changes: it
    // lights the scene (reflections and ambient) and colours the haze.
    configureSky(this.skyCopy, this.night, this.twilight);
    this.skyCopy.showSunDisc.value = 0;
    this.skyCopy.scale.setScalar(1000);
    this.capture.add(this.skyCopy);
    this.cube = new THREE.CubeRenderTarget(256, {
      type: THREE.HalfFloatType,
      generateMipmaps: true,
    });
    this.cubeCamera = new THREE.CubeCamera(1, 2000, this.cube);
    this.pmrem = new THREE.PMREMGenerator(renderer);
    this.setTime(options.time);
    this.recapture();
    this.envTarget = this.lastTarget!;
    this.environment = this.envTarget.texture;
    scene.environment = this.environment;
    scene.environmentIntensity = options.skyLight;

    const distance = positionWorld.distance(cameraPosition);
    const haze = float(1).sub(exp(distance.mul(this.hazeDensity.negate())));
    const edge = smoothstep(
      options.farDistance * 0.85,
      options.farDistance,
      distance,
    );
    const viewDir = positionWorld.sub(cameraPosition).normalize();
    // Haze takes the sky's colour a little above the horizon, where it is bluer: the band
    // right at the horizon is near white under a high sun and makes distant land look milky.
    // At the terrain's far edge it blends back to the true horizon colour, so the land's
    // silhouette doesn't show against the sky.
    const hazeDir = vec3(
      viewDir.x,
      viewDir.y.add(this.hazeLift.mul(edge.oneMinus())),
      viewDir.z,
    );
    scene.fogNode = fog(
      cubeTexture(this.cube.texture, hazeDir.normalize()),
      max(haze, edge),
    );
  }

  /**
   * How much direct sun reaches a point (0..1): marches towards the sun over the terrain
   * heights. Cheap (~75 height samples), so it can run every frame for the car.
   */
  sunVisibility(
    point: THREE.Vector3,
    heightAt: (x: number, z: number) => number,
  ): number {
    const d = this.sunDirection;
    let visible = 1;
    // Start a few metres out so the ground right at the car (banking, slope) doesn't count.
    for (let t = 6; t < 300; t += 4) {
      const clearance =
        point.y + d.y * t - heightAt(point.x + d.x * t, point.z + d.z * t);
      // Soft edge that widens with distance, like a penumbra.
      visible = Math.min(visible, clearance / (0.5 + t * 0.01));
      if (visible <= 0) return 0;
    }
    return Math.min(1, visible);
  }

  private lastTarget?: THREE.RenderTarget;

  /** Local solar time in hours. */
  get time(): number {
    return this.hours;
  }

  /** Moves the sun (or at night the moon) and the lighting to a local solar time. */
  setTime(hours: number): void {
    this.hours = ((hours % 24) + 24) % 24;
    const { elevation, azimuth } = sunAngles(this.hours);
    const rad = THREE.MathUtils.degToRad;
    this.sunDirection.setFromSphericalCoords(
      1,
      rad(90 - elevation),
      rad(azimuth),
    );
    this.sky.sunPosition.value.copy(this.sunDirection);
    this.skyCopy.sunPosition.value.copy(this.sunDirection);

    const o = this.options;
    const day = THREE.MathUtils.smoothstep(elevation, -1, 4);
    // Sunlight: warmer and dimmer towards the horizon. Moonlight takes over at night, from
    // high in the opposite part of the sky.
    const warm = THREE.MathUtils.clamp(elevation / 45, 0, 1);
    this.darkness = 1 - THREE.MathUtils.smoothstep(elevation, -8, 3);
    if (elevation > -1) {
      this.sun.color.setRGB(1, 0.8 + 0.18 * warm, 0.6 + 0.36 * warm);
      this.sun.intensity = o.sunLight * (0.7 + 0.3 * warm) * day;
      this.sun.position.copy(this.sunDirection);
    } else {
      this.sun.color.setRGB(0.6, 0.7, 1);
      this.sun.intensity =
        MOONLIGHT * THREE.MathUtils.smoothstep(-elevation, 1, 6);
      this.sun.position.setFromSphericalCoords(1, rad(50), rad(azimuth + 180));
    }
    this.sun.position.multiplyScalar(500);
    this.fill.intensity =
      o.fillLight *
      Math.max(0.04, THREE.MathUtils.smoothstep(elevation, -8, 12));
    const grade = THREE.MathUtils.smoothstep(elevation, 2, 20);
    skyGain.value = THREE.MathUtils.lerp(1, SKY_GAIN, grade);
    skySaturation.value = THREE.MathUtils.lerp(1, SKY_SATURATION, grade);
    this.night.value = 1 - THREE.MathUtils.smoothstep(elevation, -12, 0);
    this.twilight.value =
      THREE.MathUtils.smoothstep(elevation, -15, -5) *
      (1 - THREE.MathUtils.smoothstep(elevation, -2, 1));
    // The eye adapts to the light falling on the ground (sun on a level surface plus the
    // ambient), relative to the late afternoon the look is tuned for. Partial adaptation,
    // capped so that night still reads as night.
    const ref = groundLight(o, REFERENCE_ELEVATION);
    this.exposure = Math.min(
      MAX_ADAPTATION,
      (ref / groundLight(o, elevation, this.sun.intensity)) ** ADAPTATION,
    );
  }

  /** Re-renders the sky light after the time changed. Costs a cube render plus filtering. */
  private recapture(): void {
    this.cubeCamera.update(this.renderer, this.capture);
    this.lastTarget = this.pmrem.fromCubemap(
      this.cube.texture,
      this.lastTarget ?? null,
    );
    this.captured = this.hours;
  }

  /** Keeps the sky centred on the camera, and the sky light in step with the time. */
  update(camera: THREE.Camera): void {
    this.sky.position.copy(camera.position);
    if (this.captured !== this.hours) this.recapture();
  }
}

/** Far end of each shadow cascade, in metres from the camera. */
const SHADOW_SPLITS = [16, 110, 450];

/** Sun elevation (degrees) whose light the base exposure is tuned for. */
const REFERENCE_ELEVATION = 23;
/** How fully exposure follows the light (1 = completely), and its cap at night. */
const ADAPTATION = 1;
const MAX_ADAPTATION = 2.5;

/** Light falling on level ground: direct sun (or moon) plus the fill and sky ambient. */
function groundLight(
  o: AtmosphereOptions,
  elevation: number,
  direct?: number,
): number {
  const warm = THREE.MathUtils.clamp(elevation / 45, 0, 1);
  const sun =
    direct ??
    o.sunLight *
      (0.7 + 0.3 * warm) *
      THREE.MathUtils.smoothstep(elevation, -1, 4);
  const height = Math.sin(
    THREE.MathUtils.degToRad(elevation > -1 ? Math.max(elevation, 0) : 40),
  );
  const ambient =
    AMBIENT * Math.max(0.04, THREE.MathUtils.smoothstep(elevation, -8, 12));
  return sun * height + ambient;
}

/** Moonlight intensity, relative to the sun's. */
const MOONLIGHT = 0.2;
/** Ambient light on level ground in daylight, in sun-intensity units. */
const AMBIENT = 1;

/**
 * Sky grade. As modelled, the sky is so bright that tone mapping clips its blue and pales
 * it towards white; dimmed and a little more saturated it reads as a clear, deep blue. The
 * sky light that lights the land is dimmed with it, which keeps the land's shadows rich.
 * Towards sunset the sky is no longer that bright, so the grade fades out.
 */
export const skySaturation = uniform(1.3);
export const skyGain = uniform(0.35);
const SKY_SATURATION = 1.3;
const SKY_GAIN = 0.35;

function configureSky(
  sky: SkyMesh,
  night: UniformNode<"float", number>,
  twilight: UniformNode<"float", number>,
): void {
  const material = sky.material as THREE.NodeMaterial;
  const day = material.colorNode as Node<"vec4">;
  // Below the horizon the sky model goes brown-black. Night: dimmed, deep blue, lighter
  // towards the horizon. Twilight: a blue glow low in the sky, warmer towards the sun.
  const dir = positionWorld.sub(cameraPosition).normalize();
  const up = dir.y.max(0);
  const sunward = dir.xz
    .normalize()
    .dot(sky.sunPosition.xz.normalize())
    .mul(0.5)
    .add(0.5);
  const dark = vec3(0.0015, 0.003, 0.008).mul(float(3).sub(up.mul(2)));
  const glow = mix(
    vec3(0.012, 0.022, 0.05),
    vec3(0.06, 0.035, 0.018),
    sunward.pow(3),
  )
    .mul(exp(up.mul(-5)))
    .mul(twilight);
  const luma = day.rgb.dot(vec3(0.2126, 0.7152, 0.0722));
  const graded = mix(vec3(luma), day.rgb, skySaturation).max(0).mul(skyGain);
  material.colorNode = vec4(
    mix(
      graded,
      day.rgb
        .mul(vec3(0.3, 0.4, 0.8))
        .add(dark)
        .add(glow),
      night,
    ),
    1,
  );
  sky.turbidity.value = 2.5;
  sky.rayleigh.value = 2;
  sky.mieCoefficient.value = 0.004;
  sky.mieDirectionalG.value = 0.8;
  sky.cloudCoverage.value = 0.35;
  sky.cloudDensity.value = 0.35;
}
