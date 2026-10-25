import type * as THREE from "three/webgpu";

/** Render resolution choices, as a share of the display's (capped at 2x) pixel density. */
const MIN = 50;
const MAX = 100;
const STEP = 5;

/**
 * The pause menu (Escape): the last frame stays up, blurred, behind a panel with settings.
 * Resolution renders the 3D scene at a share of the display's pixel density (the HUD, being
 * HTML, stays sharp); it is the largest single lever on GPU time.
 */
export class PauseMenu {
  private readonly root = document.createElement("div");
  private readonly slider = document.createElement("input");
  private readonly value = document.createElement("output");
  private readonly resume = document.createElement("button");
  /** Display pixel density the percentage applies to. */
  private readonly native = Math.min(devicePixelRatio, 2);

  constructor(
    private readonly renderer: THREE.WebGPURenderer,
    /** Called when the menu opens or closes. */
    private readonly onToggle: (open: boolean) => void,
    /** Called after a setting changed the image (to redraw it while paused). */
    private readonly onChange: () => void,
  ) {
    this.root.id = "paused";
    this.root.hidden = true;
    const panel = document.createElement("div");
    panel.className = "menu";
    const title = document.createElement("h1");
    title.textContent = "Paused";

    const row = document.createElement("label");
    row.className = "setting";
    const name = document.createElement("span");
    name.textContent = "Resolution";
    this.slider.type = "range";
    this.slider.min = String(MIN);
    this.slider.max = String(MAX);
    this.slider.step = String(STEP);
    const saved = Number(localStorage.getItem("resolution"));
    this.slider.value = String(saved >= MIN && saved <= MAX ? saved : MAX);
    this.slider.addEventListener("input", () => this.apply(true));
    row.append(name, this.value, this.slider);

    this.resume.textContent = "Resume";
    this.resume.addEventListener("click", () => this.close());
    panel.append(title, row, this.resume);
    this.root.append(panel);
    document.body.append(this.root);

    window.addEventListener("keydown", (e) => {
      if (e.code !== "Escape" || e.repeat) return;
      if (this.open) this.close();
      else this.show();
    });
    this.apply(false);
  }

  get open(): boolean {
    return !this.root.hidden;
  }

  private show(): void {
    this.root.hidden = false;
    this.onToggle(true);
  }

  private close(): void {
    this.root.hidden = true;
    // Nothing in the menu keeps focus: Space (handbrake) would press a focused control.
    (document.activeElement as HTMLElement | null)?.blur();
    this.onToggle(false);
  }

  /** Re-applies the resolution (after the window was resized, too). */
  apply(changed = false): void {
    const percent = Number(this.slider.value);
    this.renderer.setPixelRatio((this.native * percent) / 100);
    const canvas = this.renderer.domElement;
    this.value.textContent = `${percent}%  ·  ${canvas.width} × ${canvas.height}`;
    if (changed) {
      localStorage.setItem("resolution", String(percent));
      this.onChange();
    }
  }
}
