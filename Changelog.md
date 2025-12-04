
## 2025-12-03

### Added

* New `CameraControls`, with keyboard controlled movement and mouse controlled navigation modes (Pan, Orbit, Drag).

* Handling of tiny splats which could previously lead to invalid ellipses and shrink beyond visibility.

* Handling of frustum bounds.

### Removed

* Previous orbit only camera controls.

### Fixed

* Bug in `buildTileListKernel`, if a chunk of splats was completely out of frustum, the kernel would enter an invalid state, stuck.
