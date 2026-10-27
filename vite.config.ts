import { readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { defineConfig } from "vite";

/** Sizes of the files in public/ (path -> bytes), for the loading screen's download
 * progress: images report none while they load, so their weight has to be known up front. */
function publicSizes(dir = "public"): Record<string, number> {
  const sizes: Record<string, number> = {};
  const walk = (d: string) => {
    for (const name of readdirSync(d)) {
      const path = join(d, name);
      const stat = statSync(path);
      if (stat.isDirectory()) walk(path);
      else sizes[relative(dir, path).split("\\").join("/")] = stat.size;
    }
  };
  walk(dir);
  return sizes;
}

export default defineConfig({
  build: { target: "es2023" },
  define: { __PUBLIC_SIZES__: JSON.stringify(publicSizes()) },
});
