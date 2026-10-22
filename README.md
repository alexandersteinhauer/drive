# drive

A browser driving game (Three.js + Vite + TypeScript). The app lives in `web/`.

## Develop

```sh
cd web
bun install
bun run dev        # http://localhost:5173
```

## Deploy to Cloudflare Workers

The site is fully static. `web/wrangler.jsonc` deploys the Vite build (`web/dist`) as a
Worker with static assets; no Worker script or bindings are needed.

### From your machine

```sh
cd web
bun install
bunx wrangler login     # once
bun run deploy          # typecheck, build, wrangler deploy
```

It is served at `https://drive.<your-subdomain>.workers.dev`. Change `name` in
`web/wrangler.jsonc` to rename the Worker; add a custom domain in the Cloudflare dashboard
(Workers & Pages > drive > Settings > Domains & Routes).

`bun run preview:cf` builds and serves the site locally through Wrangler.

### From GitHub (Workers Builds, deploys on every push)

In the Cloudflare dashboard: Workers & Pages > Create > Import a repository, pick this repo,
then set:

| Setting        | Value                  |
| -------------- | ---------------------- |
| Root directory | `web`                  |
| Build command  | `bun run build`        |
| Deploy command | `bunx wrangler deploy` |

The Worker name in the dashboard must match `name` in `web/wrangler.jsonc` (`drive`).
