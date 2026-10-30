/** Only the Pista opts into the learned exhaust. An explicit URL wins over saved settings. */
export type PistaExhaustMode = "original" | "surrogate" | "surrogate-retrained";

export function pistaExhaustMode(
  url: URL,
  saved: string | null,
): PistaExhaustMode {
  const selected = url.searchParams.get("exhaust") ?? saved;
  return selected === "surrogate" || selected === "surrogate-retrained"
    ? selected
    : "original";
}

/** Describe the selected model, or the actual backend once it has initialized. */
export function pistaExhaustStatus(
  mode: PistaExhaustMode,
  active?: boolean,
): string {
  if (mode === "original") return "Original exhaust active.";
  if (mode === "surrogate-retrained") {
    if (active === false)
      return "Retrained preview unavailable; original exhaust active.";
    return `Retrained pressure-only preview${active ? " active" : ""}; experimental, bass still too strong.`;
  }
  if (active === undefined)
    return "Exhaust-only pilot; known transient and training-rate limits.";
  return active
    ? "Exhaust-only pilot active; transients are experimental."
    : "Pilot unavailable; original exhaust active.";
}

/** Changing the pause-menu setting restarts in the Pista, preserving the world URL. */
export function pistaExhaustUrl(url: URL, mode: PistaExhaustMode): URL {
  const next = new URL(url);
  next.searchParams.set("exhaust", mode);
  next.searchParams.set("car", "pista");
  return next;
}
