import { SURROGATE_GLOBALS, type EngineSpec } from "../sim/engineSpec";

/** Remove only the experimental audio selector, retaining the original voice and physics. */
export function originalExhaustSpec(spec: EngineSpec): EngineSpec {
  const out = structuredClone(spec);
  delete out.audio.surrogate;
  return out;
}

/** Check the pilot's interface here; Rust validates every coefficient and kernel shape. */
export function validateSurrogateModel(
  bytes: ArrayBuffer,
  spec: EngineSpec,
): void {
  const model: unknown = JSON.parse(new TextDecoder().decode(bytes));
  if (!model || typeof model !== "object")
    throw new Error("invalid exhaust model");
  const m = model as Record<string, unknown>;
  const cylinders = spec.banks.reduce((n, b) => n + b.cylinders.length, 0);
  if (m.cylinders !== cylinders)
    throw new Error("exhaust model cylinder count does not match the engine");
  if (!Array.isArray(m.widths) || m.widths.length !== SURROGATE_GLOBALS)
    throw new Error("exhaust model must use the five live engine controls");
  // The shipped pilot predicts the two radiated exhaust pressures, not jet noise or
  // teacher-only fluid state. Never accidentally play another model's raw outputs.
  if (
    !Array.isArray(m.kernels) ||
    !Array.isArray(m.kernels[0]) ||
    m.kernels[0].length !== 2
  )
    throw new Error("exhaust pilot must predict two pressure outputs");
  if (m.rate !== 20000)
    throw new Error("exhaust pilot must run at its trained 20 kHz rate");
  // New models declare their native training rate. Do not silently recreate the old
  // train/runtime mismatch if engine settings change; legacy pilots lack this metadata.
  if (m.nativeRate !== undefined && m.nativeRate !== spec.simulationFrequency)
    throw new Error(
      "exhaust model native engine rate does not match the source",
    );
}

export type ModelRequest = (
  url: string,
  options?: RequestInit,
) => Promise<Response>;

/** Optional asset: never requested for the original exhaust or for the other cars. */
export async function fetchSurrogateModel(
  spec: EngineSpec,
  base: string,
  request: ModelRequest = fetch,
): Promise<ArrayBuffer | undefined> {
  const surrogate = spec.audio.surrogate;
  if (!surrogate) return undefined;
  const response = await request(base + surrogate.model, {
    signal: AbortSignal.timeout(15000),
  });
  if (!response.ok)
    throw new Error(`exhaust model request failed (${response.status})`);
  const bytes = await response.arrayBuffer();
  validateSurrogateModel(bytes, spec);
  return bytes;
}
