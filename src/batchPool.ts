import * as THREE from "three/webgpu";

export interface PoolSize {
  vertices: number;
  indices: number;
  instances: number;
}

/**
 * Many streamed meshes of one material as a single object: a BatchedMesh, which the renderer
 * prepares once per frame and then draws with one cheap native draw per visible piece (culled
 * and sorted per piece). Each piece gets a slot of a fixed size ("room"); freed slots are
 * reused by pieces of the same room, so streaming never moves or reallocates other data. The
 * buffers only grow (a full re-upload) when a piece finds no room, so size the pool for the
 * peak.
 */
export class BatchPool {
  readonly mesh: THREE.BatchedMesh;
  private readonly free = new Map<string, number[]>();
  private readonly slots: { room: string; instance: number }[] = [];
  private vertices = 0;
  private indices = 0;
  private instances = 0;

  constructor(
    material: THREE.Material,
    private size: PoolSize,
  ) {
    // Over 65535 vertices from the start: the index buffer is then 32-bit and stays so.
    size.vertices = Math.max(size.vertices, 65536);
    this.mesh = new THREE.BatchedMesh(
      size.instances,
      size.vertices,
      size.indices,
      material,
    );
    // Culled per piece instead.
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
  }

  /**
   * Adds a piece (indexed, with the pool's attributes and a bounding sphere), in a slot of
   * `vertices` x `indices` (by default exactly its size). Returns its id.
   */
  add(
    geometry: THREE.BufferGeometry,
    vertices = geometry.getAttribute("position").count,
    indices = geometry.getIndex()!.count,
  ): number {
    const room = `${vertices},${indices}`;
    const reuse = this.free.get(room)?.pop();
    if (reuse !== undefined) {
      this.mesh.setGeometryAt(reuse, geometry);
      this.mesh.setVisibleAt(this.slots[reuse]!.instance, true);
      return reuse;
    }
    this.reserve(vertices, indices);
    const id = this.mesh.addGeometry(geometry, vertices, indices);
    this.slots[id] = { room, instance: this.mesh.addInstance(id) };
    return id;
  }

  /** Hides a piece and frees its slot for the next piece of the same room. */
  remove(id: number): void {
    const slot = this.slots[id]!;
    this.mesh.setVisibleAt(slot.instance, false);
    const list = this.free.get(slot.room);
    if (list) list.push(id);
    else this.free.set(slot.room, [id]);
  }

  private reserve(vertices: number, indices: number): void {
    this.vertices += vertices;
    this.indices += indices;
    this.instances++;
    const s = this.size;
    if (this.vertices > s.vertices || this.indices > s.indices) {
      s.vertices = Math.max(s.vertices, this.vertices) * 1.5;
      s.indices = Math.max(s.indices, this.indices) * 1.5;
      if (import.meta.env.DEV)
        console.warn(
          `BatchPool ${this.mesh.name}: growing to ${s.vertices} vertices, ${s.indices} indices`,
        );
      this.mesh.setGeometrySize(Math.ceil(s.vertices), Math.ceil(s.indices));
    }
    if (this.instances > s.instances) {
      s.instances *= 2;
      if (import.meta.env.DEV)
        console.warn(
          `BatchPool ${this.mesh.name}: growing to ${s.instances} instances`,
        );
      this.mesh.setInstanceCount(s.instances);
    }
  }
}
