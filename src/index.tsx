/* @refresh reload */
import { render } from 'solid-js/web'
import App from './App.tsx'

const root = document.getElementById('root')

// @ts-expect-error(js sucks)
render(() => <App />, root!)
