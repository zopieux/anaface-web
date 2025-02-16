import type { DetectedFaces, InitOutput } from 'anaface-rs'
import type { InferenceSession } from 'onnxruntime-web/all'
import type { Command, Kpss, RawDetectedFace, RFaces, RReady } from './util'

import retinaface from 'anaface-rs'

import modelPath from './assets/det_10g.onnx?url'

import { relativeBbox } from './util'

/*
 * Do not import anything from onnxruntime-web (beyond types) in root.
 * This breaks ort's wasm threading for some ungodly cursed reason.
 */

interface State {
  session?: InferenceSession
  canvas?: OffscreenCanvas
  ctx?: OffscreenCanvasRenderingContext2D
  retinaface?: InitOutput
}

const state: State = {}

async function setupOrt() {
  if (globalThis.ort !== undefined)
    return
  performance.mark('setup-ort')
  const ort = await import('onnxruntime-web/all')
  ort.env.wasm.proxy = false
  ort.env.wasm.numThreads = 4
  globalThis.ort = ort
  logDuration('setup-ort')
}

globalThis.onmessage = async function (ev: MessageEvent<Command>) {
  switch (ev.data.cmd) {
    case 'init':
      await init(ev.data.canvas)
      break
    case 'analyze':
      await detectFaces(ev.data.image)
      break
    default:
      console.error('[anaface/worker] received unexpected command', ev.data)
  }
}

async function init(canvas: OffscreenCanvas) {
  if (canvas!.width !== canvas!.height)
    throw new Error('non square canvas')

  await setupOrt()

  performance.mark('init')
  const [rf, session] = await Promise.all([retinaface(), globalThis.ort.InferenceSession.create(modelPath, {
    graphOptimizationLevel: 'all',
    executionProviders: ['webnn', 'webgpu', 'cuda', 'wasm'],
  })])
  logDuration('init')
  state.canvas = canvas
  state.ctx = state.canvas.getContext('2d', { willReadFrequently: true })!
  state.retinaface = rf
  state.session = session
  globalThis.postMessage(<RReady>{ what: 'ready' })
}

async function detectFaces(img: ImageBitmap): Promise<void> {
  const size = state.canvas!.width
  const factor = Math.min(1, Math.min(size / img.width, size / img.height))
  performance.mark('paint')
  const ctx = state.ctx!
  ctx.fillStyle = 'black'
  ctx.fillRect(0, 0, size, size)
  ctx.save()
  if (factor !== 1)
    ctx.scale(factor, factor)
  ctx.drawImage(img, 0, 0)
  ctx.restore()
  logDuration('paint')
  performance.mark('pixels')
  // const data = ctx.getImageData(0, 0, size, size).data
  // const array = state.retinaface!.get_bgr_pixels_3_height_width(data, size, size)
  const array = getBGRPixels3HeighWidth(ctx, size, size)
  logDuration('pixels')
  performance.mark('run')
  const { Tensor } = await import('onnxruntime-web/all')
  const outs = (await state.session!.run({
    'input.1': new Tensor('float32', array, [1, 3, size, size]),
  }))
  logDuration('run')
  const outputs = state.session!.outputNames.map(name => outs[name])
  const [s8, s16, s32, b8, b16, b32, k8, k16, k32] = outputs.map(o => o.data as Float32Array)
  const ratio = img.width / img.height
  let w, h
  if (ratio > 1) {
    w = size / ratio
    h = size
  }
  else {
    w = size
    h = size * ratio
  }
  const det_scale = h / w
  performance.mark('retinaface')
  const detectedFaces: DetectedFaces = state.retinaface?.retinaface(s8, s16, s32, b8, b16, b32, k8, k16, k32, size, det_scale, 0.5, 0.4)
  logDuration('retinaface')
  const faces = detectedFaces.map(f => <RawDetectedFace>{
    score: f.detection_score,
    bbox: relativeBbox(f.bbox, img.width * factor, img.height * factor),
    kpss: f.kpss as unknown as Kpss,
  })
  globalThis.postMessage(<RFaces>{ what: 'faces', faces })
}

function getBGRPixels3HeighWidth(ctx: OffscreenCanvasRenderingContext2D, width: number, height: number): Float32Array {
  const MEAN = 127.5
  const STD = 128.0
  const wh = width * height
  const data = ctx.getImageData(0, 0, width, height).data
  const rgbArray = new Float32Array(width * height * 3)
  for (let y = 0; y < height; y++) {
    const yw = y * width
    for (let x = 0; x < width; x++) {
      const index = (yw + x) * 4
      rgbArray[0 * wh + yw + x] = (data[index + 2] - MEAN) / STD
      rgbArray[1 * wh + yw + x] = (data[index + 1] - MEAN) / STD
      rgbArray[2 * wh + yw + x] = (data[index + 0] - MEAN) / STD
    }
  }
  return rgbArray
}

function logDuration(mark: string) {
  const d = performance.measure(`${mark}-duration`, mark).duration
  console.debug(`[anaface/perf] ${mark} took ${d.toFixed(0)}ms`)
}
