# Model credits

## bmw-i4.glb

"2022 BMW i4" by tonielpro520, licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).
Source: https://sketchfab.com/3d-models/2022-bmw-i4-1f23525ab55746899e3645e3352981c2

Changes: interior and licence plate removed; scaled, reoriented and re-centred; wheel geometry
split per corner; meshes merged, simplified (387k to 259k triangles) and compressed; textures
resized and converted to WebP. Generated with `scripts/prepare-car.ts`.

BMW and i4 are trademarks of BMW AG. This project is not affiliated with or endorsed by BMW.

## nature.glb

Trees and bushes generated with EZ-Tree by Daniel Greenheck, MIT licence
(https://github.com/dgreenheck/ez-tree, `@dgreenheck/ez-tree` 1.1.0), from its oak, ash, pine
and bush presets with fixed seeds. Bark textures shipped with EZ-Tree: oak from Poly Haven
(https://polyhaven.com/a/bark_brown_02, CC0), pine from TextureCan (https://www.texturecan.com/details/588/,
CC0); leaf textures from EZ-Tree (MIT).

Grass, clover, ferns and rocks from "Stylized Nature MegaKit" (Standard, free version) by
Quaternius, released under CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/).
Source: https://quaternius.com/packs/stylizednaturemegakit.html

Changes: normalised to metres, leaf normals bent outwards from each crown, two levels of detail
per tree (simplified, small branches dropped, leaf cards thinned out), textures resized and
converted to WebP, meshes compressed. Generated with `scripts/prepare-nature.ts`. Distant trees
are billboards rendered from these models at load time.
