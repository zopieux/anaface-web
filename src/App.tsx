import type { CAnalyze, CInit, RawDetectedFace, Reply } from './util'
import { batch, createEffect, createSignal, on, onMount, Show } from 'solid-js'

import h from 'solid-js/h'
import { RelativeBbox } from './util'
import OnnxWorker from './worker?worker'
import './App.sass'

// import testImage from '/faces_h.jpg?url'

const LOAD_TEST_IMAGE = false
const INFER_SIZE = 640

function setDragging(dragging: boolean) {
  return (ev: DragEvent) => (ev.target as HTMLInputElement).classList.toggle('dragging', dragging)
}

function App() {
  const worker = new OnnxWorker()

  let displaySvg!: SVGSVGElement

  const [isReady, setReady] = createSignal(false)
  const [image, setImage] = createSignal<ImageBitmap>()
  const [blob, setBlob] = createSignal<Blob>()
  const [faces, setFaces] = createSignal<RawDetectedFace[]>([])

  worker.onmessage = async (ev: MessageEvent<Reply>) => {
    switch (ev.data.what) {
      case 'ready':
        setReady(true)
        break
      case 'faces':
        setFaces(ev.data.faces as RawDetectedFace[])
        break
    }
  }

  async function handleFile(event: Event) {
    const file = (event.target! as HTMLInputElement).files![0]
    const image = await createImageBitmap(file)
    batch(() => {
      setFaces([])
      setBlob(file)
      setImage(image)
    })
  }

  const transferCanvas = (el: HTMLCanvasElement) => {
    const canvas = el.transferControlToOffscreen()
    worker.postMessage({ cmd: 'init', canvas } as CInit, [canvas])
  }

  createEffect(on([isReady, image], ([ready, image]) => {
    if (ready && image)
      worker.postMessage({ cmd: 'analyze', image } as CAnalyze)
  }, { defer: true }))

  const faceSvgRect = (f: RawDetectedFace) => {
    const im = image()!
    const [x, y, x2, y2] = new RelativeBbox(f.bbox).forSize(im.width, im.height)
    const [ww, hh] = [x2 - x, y2 - y]
    return (
      <g transform={`translate(${x} ${y})`}>
        <foreignObject width={ww} height={hh} style={{ 'border-radius': '3px', 'overflow': 'hidden' }}>
          <div style={{ 'display': 'flex', 'width': '100%', 'height': '100%', 'justify-content': 'flex-end', 'align-items': 'flex-start', 'font-family': 'sans-serif' }}>
            <span style={{ 'display': 'flex', 'background': 'rgba(255,255,255,50%)', 'padding': '0 3px', 'border-radius': '0 0 0 3px' }}>{(f.score * 100).toFixed(0)}</span>
          </div>
        </foreignObject>
        <rect fill="none" stroke="red" stroke-width={2} x={0} y={0} width={ww} height={hh} rx={3} ry={3} />
      </g>
    )
  }

  const imageHref = () => {
    return blob() ? URL.createObjectURL(blob()!) : undefined
  }

  let fileInput!: HTMLInputElement
  onMount(async () => {
    if (LOAD_TEST_IMAGE) {
      // const arrayBuffer = await fetch(testImage).then(b => b.arrayBuffer())
      // const dataTransfer = new DataTransfer()
      // dataTransfer.items.add(new File([arrayBuffer], 'test.jpeg', { type: 'image/jpeg' }))
      // fileInput.files = dataTransfer.files
      // fileInput.dispatchEvent(new Event('change'))
    }
  })

  return [
    h('main', {}, [
      h('h1', 'anaface'),
      <Show when={isReady} fallback={<p>Loading…</p>}>
        <input
          type="file"
          onChange={handleFile}
          ref={fileInput}
          onDragEnter={setDragging(true)}
          onDragLeave={setDragging(false)}
          onDrop={setDragging(false)}
        />
      </Show>,
      h('div', { style: 'margin: 1em 0' }, [
        <svg ref={displaySvg} width="100%" height="100%" style={{ 'max-height': '50vh' }} viewBox={`0 0 ${image()?.width ?? 0} ${image()?.height ?? 0}`}>
          <g><image href={imageHref()} /></g>
          <g>{faces().map(faceSvgRect)}</g>
        </svg>,
        h('canvas', { width: INFER_SIZE, height: INFER_SIZE, style: 'display:none', ref: transferCanvas }),
      ]),
    ]),
  ]
}

export default App
