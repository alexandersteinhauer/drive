/** Only the Pista opts into the learned exhaust. An explicit URL wins over saved settings. */
export type PistaExhaustMode = "original" | "surrogate";

export function pistaExhaustMode(
  url: URL,
  saved: string | null,
): PistaExhaustMode {
  const selected = url.searchParams.get("exhaust");
  return (selected ?? saved) === "surrogate" ? "surrogate" : "original";
}

/** Changing the pause-menu setting restarts in the Pista, preserving the world URL. */
export function pistaExhaustUrl(url: URL, mode: PistaExhaustMode): URL {
  const next = new URL(url);
  next.searchParams.set("exhaust", mode);
  next.searchParams.set("car", "pista");
  return next;
}
