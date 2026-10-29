import type { Road } from "./world/road";
import { Minimap } from "./minimap";

export type Assist = "abs" | "esc";
export type AssistMode = "on" | "sport" | "off";

/** Click order per button: ABS is only on or off, stability control also has sport. */
const CYCLE: Record<Assist, AssistMode[]> = {
  abs: ["on", "off"],
  esc: ["on", "sport", "off"],
};
const LABEL: Record<Assist, string> = {
  abs: "ABS",
  esc: "Stability control",
};

// Dashboard-style symbols, drawn in currentColor.
const ICON: Record<Assist, string> = {
  abs: `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round">
    <circle cx="24" cy="24" r="14"/>
    <path d="M9 12a20 20 0 0 0 0 24M39 12a20 20 0 0 1 0 24"/>
    <text x="24" y="27.6" text-anchor="middle" fill="currentColor" stroke="none"
      font-family="ui-sans-serif, system-ui, sans-serif" font-size="9.5" font-weight="700">ABS</text>
  </svg>`,
  // The standard stability control symbol: a car over two skid marks.
  esc: `<svg viewBox="230 -55 900 900" fill="currentColor">
    <path fill-rule="evenodd" d="M600 100H800Q825 100 838 120L875 180H925Q938 180 938 195V212Q938 225 925 225H905L920 260V410Q920 430 900 430H855Q835 430 835 410V385H565V410Q565 430 545 430H500Q480 430 480 410V260L495 225H475Q462 225 462 212V195Q462 180 475 180H525L562 120Q575 100 600 100Z
      M610 145H790L822 180H578Z
      M550 228H850L865 248H800V295H878V340H522V295H600V248H535Z"/>
    <path id="skid" fill="none" stroke="currentColor" stroke-width="44"
      d="M510 468C580 475 650 500 615 535C575 575 395 565 392 608C390 645 480 668 565 690"/>
    <use href="#skid" x="355"/>
  </svg>`,
};

/**
 * Heads-up display along the bottom centre: speed and distance driven on the left, the minimap in the middle,
 * and the ABS and stability control buttons on the right. Each button shows its mode by
 * colour (green on, blue sport, red off); clicking cycles through the modes.
 */
export class Hud {
  readonly minimap: Minimap;
  private readonly root = document.createElement("div");
  private readonly speed = document.createElement("div");
  private readonly buttons = {} as Record<Assist, HTMLButtonElement>;
  private readonly odometer = document.createElement("div");
  private shownSpeed = -1;
  private shownTenths = -1;

  constructor(
    road: Road,
    private readonly modes: Record<Assist, AssistMode>,
    private readonly onChange: (assist: Assist, mode: AssistMode) => void,
  ) {
    this.root.id = "hud";
    this.speed.className = "speed";
    this.speed.innerHTML = `<span>0</span><small>km/h</small>`;
    this.odometer.className = "odometer";
    this.odometer.title = "Distance driven";
    this.speed.append(this.odometer);
    this.setDistance(0);
    this.root.append(this.speed);
    this.minimap = new Minimap(road, this.root);
    const assists = document.createElement("div");
    assists.className = "assists";
    for (const assist of ["abs", "esc"] as const) {
      const button = document.createElement("button");
      button.tabIndex = -1;
      button.innerHTML = ICON[assist];
      // Don't take focus: Space (handbrake) would press a focused button.
      button.addEventListener("mousedown", (e) => e.preventDefault());
      button.addEventListener("click", () => {
        const cycle = CYCLE[assist];
        const next =
          cycle[(cycle.indexOf(this.modes[assist]) + 1) % cycle.length]!;
        this.set(assist, next);
        this.onChange(assist, this.modes[assist]);
      });
      this.buttons[assist] = button;
      assists.append(button);
      this.set(assist, modes[assist]);
    }
    this.root.append(assists);
    document.body.append(this.root);
  }

  private set(assist: Assist, mode: AssistMode): void {
    this.modes[assist] = mode;
    const button = this.buttons[assist];
    button.dataset.mode = mode;
    button.title = `${LABEL[assist]}: ${mode} (click to change)`;
  }

  /** Shown or hidden (as during a crash). */
  set visible(on: boolean) {
    this.root.style.visibility = on ? "" : "hidden";
  }

  /** `speed` in m/s. */
  setSpeed(speed: number): void {
    const kmh = Math.round(Math.abs(speed) * 3.6);
    if (kmh === this.shownSpeed) return;
    this.shownSpeed = kmh;
    this.speed.firstElementChild!.textContent = String(kmh);
  }

  /** Distance driven in metres, shown in km to one decimal (truncated, like an odometer). */
  setDistance(metres: number): void {
    const tenths = Math.floor(metres / 100);
    if (tenths === this.shownTenths) return;
    this.shownTenths = tenths;
    this.odometer.textContent = `${(tenths / 10).toFixed(1)} km`;
  }
}

/** Height of a storey (m), for the fall the crash compares to. */
const STOREY = 3;

/**
 * Shown on the black screen after a fatal crash, putting the impact in terms a body knows:
 * the fall that ends at the same speed.
 */
export class CrashCard {
  private readonly root = document.createElement("div");

  constructor() {
    this.root.id = "crash";
    document.body.append(this.root);
  }

  /** Shows the card for an impact at `speed` m/s. */
  show(speed: number): void {
    const fall = (speed * speed) / (2 * 9.81);
    const floor = Math.round(fall / STOREY);
    const at = `At ${Math.round(speed * 3.6)} km/h it's like falling`;
    const like =
      floor >= 1
        ? `${at} from the ${ordinal(floor)} floor.`
        : `${at} ${fall.toFixed(0)} m.`;
    this.root.innerHTML = `<strong>You crashed</strong><span>${like}</span>`;
    this.root.classList.add("shown");
  }

  /** Adds the prompt to drive again. */
  prompt(): void {
    const hint = document.createElement("em");
    hint.textContent = "Press any key to drive again";
    this.root.append(hint);
  }

  hide(): void {
    this.root.classList.remove("shown");
  }
}

/** 1st, 2nd, 3rd, 4th, ..., 11th, 12th, 13th, ..., 21st. */
function ordinal(n: number): string {
  const teen = n % 100 >= 11 && n % 100 <= 13;
  const suffix = teen ? "th" : (["th", "st", "nd", "rd"][n % 10] ?? "th");
  return `${n}${suffix}`;
}
