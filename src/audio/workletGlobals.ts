// What the Emscripten glue (bundled by Vite) expects of a global scope that an
// AudioWorkletGlobalScope lacks. Imported first, so it runs before the glue.
const g = globalThis as {
  self?: unknown;
  location?: unknown;
  performance?: unknown;
};
g.self ??= globalThis;
g.location ??= { href: "" };
// The simulator times itself with steady_clock -> performance.now().
g.performance ??= { now: () => Date.now() };
