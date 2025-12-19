
## CURRENT

### Added

* Optional evaluation of view dependent (degree 1+) Spherical Harmonics.

### Changed

* Refactored `PlyParser` so that it does not assume a fixed set of properties anymore and can handle varying degrees of spherical harmonics.

* Dynamic reallocation of the tile list.

### Removed

* Conversion script `convert_ply.py` and its reference in the documentation, since the parser is not expecting a hardcoded set of properties anymore.

## 2025-12-03

### Added

* New `CameraControls`, with keyboard controlled movement and mouse controlled navigation modes (Pan, Orbit, Drag).

* Handling of tiny splats which could previously lead to invalid ellipses and shrink beyond visibility.

* Handling of frustum bounds.

### Removed

* Previous orbit only camera controls.

### Fixed

* Bug in `buildTileListKernel`, if a chunk of splats was completely out of frustum, the kernel would enter an invalid state, stuck.
