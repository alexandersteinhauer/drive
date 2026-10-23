/**
 * GPU benchmark in headless Chrome (real GPU, no window):
 *   bun run dev            # in another terminal
 *   bun run bench [suite] [url]    # default http://localhost:5173/; suite e.g. "trees"
 *
 * Opens the game with ?bench&seed=1 (see src/bench.ts) at 1512x804 CSS px and device
 * pixel ratio 2, like a 14" MacBook, and prints the per-pass and per-part GPU times.
 * CHROME overrides the browser binary.
 */
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const args = process.argv.slice(2);
const base = args.find((a) => a.includes("://")) ?? "http://localhost:5173/";
/** A suite of variants other than the default (see src/bench.ts), e.g. "trees". */
const suite = args.find((a) => !a.includes("://")) ?? "";
const url = new URL(base);
url.searchParams.set("seed", "1");
url.searchParams.set("bench", suite);
const chrome =
  process.env.CHROME ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const port = 9400 + Math.floor(Math.random() * 500);
const profile = mkdtempSync(join(tmpdir(), "bench-chrome-"));
const proc = spawn(
  chrome,
  [
    "--headless=new",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    "--window-size=1512,804",
    "--force-device-scale-factor=2",
    // Full-precision GPU timestamps (Chrome otherwise rounds them to ~0.1 ms).
    "--enable-dawn-features=allow_unsafe_apis",
    "--enable-webgpu-developer-features",
    "--no-first-run",
    "--no-default-browser-check",
    "about:blank",
  ],
  { stdio: "ignore" },
);
const done = () => {
  proc.kill();
  rmSync(profile, { recursive: true, force: true });
};

try {
  let version: { webSocketDebuggerUrl: string } | undefined;
  for (let i = 0; i < 50 && !version; i++) {
    await Bun.sleep(200);
    version = await fetch(`http://127.0.0.1:${port}/json/version`)
      .then((r) => r.json() as Promise<{ webSocketDebuggerUrl: string }>)
      .catch(() => undefined);
  }
  if (!version) throw new Error("Chrome did not start");
  const ws = new WebSocket(version.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r));
  let id = 0;
  const waiting = new Map<number, (v: unknown) => void>();
  ws.addEventListener("message", (e) => {
    const m = JSON.parse(String(e.data)) as {
      id?: number;
      result?: unknown;
      method?: string;
      params?: { type: string; args: { value?: unknown }[] };
    };
    if (m.id && waiting.has(m.id)) {
      waiting.get(m.id)!(m.result);
      waiting.delete(m.id);
    } else if (
      m.method === "Runtime.consoleAPICalled" &&
      m.params!.type !== "log"
    )
      console.error(
        `[page ${m.params!.type}]`,
        m.params!.args.map((a) => a.value).join(" "),
      );
  });
  const send = <T>(
    method: string,
    params: object = {},
    sessionId?: string,
  ): Promise<T> =>
    new Promise((r) => {
      const i = ++id;
      waiting.set(i, r as (v: unknown) => void);
      ws.send(JSON.stringify({ id: i, method, params, sessionId }));
    });
  const { targetId } = await send<{ targetId: string }>("Target.createTarget", {
    url: "about:blank",
  });
  const { sessionId } = await send<{ sessionId: string }>(
    "Target.attachToTarget",
    { targetId, flatten: true },
  );
  await send("Runtime.enable", {}, sessionId);
  await send("Page.navigate", { url: url.href }, sessionId);
  console.error(`benchmarking ${url.href} ...`);
  const started = Date.now();
  for (;;) {
    await Bun.sleep(1000);
    const r = await send<{ result?: { value?: string } }>(
      "Runtime.evaluate",
      { expression: "window.benchResult", returnByValue: true },
      sessionId,
    );
    if (r.result?.value) {
      console.log(r.result.value);
      break;
    }
    if (Date.now() - started > 10 * 60_000) throw new Error("timed out");
  }
  ws.close();
} finally {
  done();
}
