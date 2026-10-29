# Credits

This project's own code is licensed under the [MIT License](LICENSE). The third-party code,
models and textures it uses keep their own licences, listed below.

## Models (`public/models/`)

### bmw-i4.glb

"2022 BMW i4" by tonielpro520, licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).
Source: https://sketchfab.com/3d-models/2022-bmw-i4-1f23525ab55746899e3645e3352981c2

Changes: interior and licence plate removed; scaled, reoriented and re-centred; wheel geometry
split per corner; meshes merged, simplified (387k to 259k triangles) and compressed; textures
resized and converted to WebP. Generated with `scripts/prepare-car.ts`.

BMW and i4 are trademarks of BMW AG. This project is not affiliated with or endorsed by BMW.

### vw-beetle.glb

"1963 Volkswagen Beetle: Junkyard Herbie" by angemturc, licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).
Source: https://sketchfab.com/3d-models/1963-volkswagen-beetle-junkyard-herbie-e7d4acf212b4453e86caa55fcaa9718a

Changes: scaled, reoriented and re-centred; wheels split per corner; the shared tyre, glass and
underbody material split into an opaque and a transparent one; meshes merged and compressed;
textures padded outside their UV islands, resized and converted to WebP. Generated with
`scripts/prepare-car.ts`.

Volkswagen and Beetle are trademarks of Volkswagen AG. This project is not affiliated with or
endorsed by Volkswagen.

### nature.glb

Trees and bushes generated with EZ-Tree by Daniel Greenheck, MIT licence
(https://github.com/dgreenheck/ez-tree, `@dgreenheck/ez-tree` 1.1.0), from its oak, ash, pine
and bush presets with fixed seeds. Bark textures shipped with EZ-Tree: oak from Poly Haven
(https://polyhaven.com/a/bark_brown_02, CC0), pine from TextureCan (https://www.texturecan.com/details/588/,
CC0); leaf textures from EZ-Tree (MIT).

Grass, clover, ferns and rocks from "Stylized Nature MegaKit" (Standard, free version) by
Quaternius, released under CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/).
Source: https://quaternius.com/packs/stylizednaturemegakit.html

Changes: normalised to metres, leaf normals bent outwards from each crown, two levels of detail
per tree (simplified, small branches dropped, leaf cards thinned out), generated trees cut to
under ~1,000 triangles with their leaf textures rebuilt as clumps of sixteen sprigs (four for
bushes) on few, large cards, textures resized and converted to WebP, meshes compressed. Generated with `scripts/prepare-nature.ts`. Distant trees
are billboards rendered from these models at load time.

## Textures (`public/textures/`)

All textures are from ambientCG (https://ambientcg.com), licensed CC0 1.0, converted to WebP.

| Files               | Asset                                                  | Use                   |
| ------------------- | ------------------------------------------------------ | --------------------- |
| `asphalt015_*.webp` | [Asphalt015](https://ambientcg.com/view?id=Asphalt015) | road surface (2K)     |
| `grass004_*.webp`   | [Grass004](https://ambientcg.com/view?id=Grass004)     | meadow (1K)           |
| `ground103_*.webp`  | [Ground103](https://ambientcg.com/view?id=Ground103)   | dirt on slopes, verge |
| `rock030_*.webp`    | [Rock030](https://ambientcg.com/view?id=Rock030)       | rock in cuttings      |
| `gravel040_*.webp`  | [Gravel040](https://ambientcg.com/view?id=Gravel040)   | road shoulders, verge |

Terrain layers ship colour and normal maps only; roughness is set per layer in the shader.

## Code

### engine-sim

The engine sound is simulated by [engine-sim](https://github.com/ange-yaghi/engine-sim) by
Ange Yaghi, with its script compiler [piranha](https://github.com/ange-yaghi/piranha) and
[simple-2d-constraint-solver](https://github.com/ange-yaghi/simple-2d-constraint-solver) (all
MIT), compiled to WebAssembly (`public/audio/engine-sim.wasm`, `src/audio/engine-sim.mjs`) by
`engine-sim/build.sh`, with small changes in `engine-sim/engine-sim.patch`. The wasm embeds
engine-sim's script library and exhaust impulse responses. Its licence:

```
Copyright 2022 AngeTheGreat (Ange Yaghi)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### three.js

[three.js](https://threejs.org) (`three` 0.186) is bundled into the built app. Its licence:

```
The MIT License

Copyright © 2010-2026 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### EZ-Tree

[EZ-Tree](https://github.com/dgreenheck/ez-tree) (`@dgreenheck/ez-tree` 1.1.0) generates the
trees and bushes in `nature.glb` at build time, and its leaf textures ship inside that file.
Its licence:

```
MIT License

Copyright (c) 2024 Daniel Greenheck

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
