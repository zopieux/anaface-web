type FixedSizeArray<N extends number, T> = N extends 0 ? never[] : {
  0: T
  length: N
} & ReadonlyArray<T>

export type Bbox = FixedSizeArray<4, number>
export type Kpss = FixedSizeArray<10, number>

export interface CInit {
  cmd: 'init'
  canvas: OffscreenCanvas
}
export interface CAnalyze {
  cmd: 'analyze'
  image: ImageBitmap
}
export type Command = CInit | CAnalyze

export interface RReady { what: 'ready' }
export interface RFaces {
  what: 'faces'
  faces: RawDetectedFace[]
}
export type Reply = RReady | RFaces

export interface DetectedRelativeFace {
  score: number
  bbox: RelativeBbox
  kpss: FixedSizeArray<10, number>
}

export class RelativeBbox {
  private relBbox: Bbox

  constructor(relBbox: Bbox) {
    this.relBbox = relBbox
  }

  public forSize(width: number, height: number): [number, number, number, number] {
    const [x1, y1, x2, y2] = this.relBbox
    return [x1 * width, y1 * height, x2 * width, y2 * height]
  }

  public debugStroke(ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D) {
    const [x1, y1, x2, y2] = this.forSize(ctx.canvas.width, ctx.canvas.height)
    ctx.strokeRect(x1, y1, (x2 - x1), (y2 - y1))
  }
}

export interface RawDetectedFace {
  score: number
  bbox: Bbox
  kpss: Kpss
}

export function relativeBbox([x1, y1, x2, y2]: [number, number, number, number], width: number, height: number): Bbox {
  return [x1 / height, y1 * width / height / height, x2 / height, y2 * width / height / height]
}
