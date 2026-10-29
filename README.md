# drive

A relaxing endless drive through rolling hills and forests, right in your browser.

**Play at [drive.st1.io](https://drive.st1.io)**

![Drifting through a corner in the hills on a sunny afternoon](.github/screenshot.jpg)

Drive with WASD or the arrow keys, pause with Escape, switch cars with C. The VW Beetle's
engine is simulated piston by piston, and it is what you hear: see [engine/](engine/).

## Run locally

```sh
bun install
bun run dev
```

The engine simulation is Rust compiled to WebAssembly; the built `src/sim/engine.wasm` is
checked in. To rebuild it: `rustup target add wasm32-unknown-unknown`, then
`bun run build-engine`.

## License

[MIT](LICENSE). Third-party models and textures keep their own licences; see
[CREDITS.md](CREDITS.md).
