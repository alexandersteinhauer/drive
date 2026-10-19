import * as THREE from "three/webgpu";
import {
  cameraPosition,
  cubeTexture,
  exp,
  float,
  fog,
  max,
  positionWorld,
  smoothstep,
} from "three/tsl";
import { CSMShadowNode } from "three/addons/csm/CSMShadowNode.js";
import { SkyMesh } from "three/addons/objects/SkyMesh.js";

export interface AtmosphereOptions {
  /**
   * Sun elevation and azimuth in degrees, as seen from the scene. Azimuth 0 is behind the
   * road's main direction (+Z), 90 is to the right (+X).
   */
  elevation: number;
  azimuth: number;
  /** Haze: half the light is scattered away after this many metres. */
  hazeHalfDistance: number;
  /** Everything is fully hazed out by this distance (hides the terrain edge). */
  farDistance: number;
  /** Sun intensity at high elevation, and the strength of the sky's image-based light. */
  sunLight: number;
  skyLight: number;
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

  constructor(
    renderer: THREE.WebGPURenderer,
    scene: THREE.Scene,
    options: AtmosphereOptions,
  ) {
    const phi = THREE.MathUtils.degToRad(90 - options.elevation);
    const theta = THREE.MathUtils.degToRad(options.azimuth);
    this.sunDirection.setFromSphericalCoords(1, phi, theta);
    configureSky(this.sky, this.sunDirection);
    this.sky.scale.setScalar(2000);
    scene.add(this.sky);

    // Sun light: warmer and dimmer towards the horizon.
    const warm = THREE.MathUtils.clamp(options.elevation / 45, 0, 1);
    this.sun.color.setRGB(1, 0.86 + 0.12 * warm, 0.72 + 0.24 * warm);
    this.sun.intensity = options.sunLight * (0.7 + 0.3 * warm);
    this.sun.position.copy(this.sunDirection).multiplyScalar(500);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0003;
    this.sun.shadow.normalBias = 0.05;
    // Cascaded shadows: sharp near the car, still present on hills hundreds of metres away.
    const csm = new CSMShadowNode(this.sun, {
      cascades: 3,
      maxFar: 450,
      mode: "practical",
      lightMargin: 150,
    });
    csm.fade = true;
    this.sun.shadow.shadowNode = csm;
    scene.add(this.sun, this.sun.target);

    // The sky without its sun disc, captured once into a cube map: it lights the scene
    // (reflections and ambient) and colours the haze.
    const capture = new THREE.Scene();
    const skyCopy = new SkyMesh();
    configureSky(skyCopy, this.sunDirection);
    skyCopy.showSunDisc.value = 0;
    skyCopy.cloudCoverage.value = this.sky.cloudCoverage.value;
    skyCopy.scale.setScalar(1000);
    capture.add(skyCopy);
    this.cube = new THREE.CubeRenderTarget(256, {
      type: THREE.HalfFloatType,
      generateMipmaps: true,
    });
    new THREE.CubeCamera(1, 2000, this.cube).update(renderer, capture);
    this.environment = new THREE.PMREMGenerator(renderer).fromCubemap(
      this.cube.texture,
    ).texture;
    scene.environment = this.environment;
    scene.environmentIntensity = options.skyLight;

    const distance = positionWorld.distance(cameraPosition);
    const haze = float(1).sub(
      exp(distance.mul(-Math.LN2 / options.hazeHalfDistance)),
    );
    const edge = smoothstep(
      options.farDistance * 0.75,
      options.farDistance,
      distance,
    );
    const viewDir = positionWorld.sub(cameraPosition).normalize();
    scene.fogNode = fog(
      cubeTexture(this.cube.texture, viewDir),
      max(haze, edge),
    );
  }

  /** Keeps the sky centred on the camera. */
  update(camera: THREE.Camera): void {
    this.sky.position.copy(camera.position);
  }
}

function configureSky(sky: SkyMesh, sun: THREE.Vector3): void {
  sky.turbidity.value = 2.5;
  sky.rayleigh.value = 2;
  sky.mieCoefficient.value = 0.004;
  sky.mieDirectionalG.value = 0.8;
  sky.cloudCoverage.value = 0.35;
  sky.cloudDensity.value = 0.35;
  sky.sunPosition.value.copy(sun);
}
