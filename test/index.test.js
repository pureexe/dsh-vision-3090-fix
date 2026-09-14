import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Config, apply } from '../src/index.js'

/** A minimal ctx double: runs the effect immediately and remembers its disposer. */
function fakeCtx() {
  let dispose
  return {
    effect: (fn) => { dispose = fn() },
    stop: () => dispose?.(),
  }
}

test('Config: models defaults to empty (caps every model)', () => {
  const config = Config({ upstreamOrigin: 'http://127.0.0.1:1234', listenPort: 8931 })
  assert.deepEqual(config.models, [])
  assert.equal(config.maxImagesPerRequest, 1)
})

test('Config: models can scope the cap to specific ids', () => {
  const config = Config({ upstreamOrigin: 'http://127.0.0.1:1234', listenPort: 8931, models: ['qwen3.8-27b'] })
  assert.deepEqual(config.models, ['qwen3.8-27b'])
})

test('Config requires upstreamOrigin and listenPort', () => {
  assert.throws(() => Config({ listenPort: 8931 }))
  assert.throws(() => Config({ upstreamOrigin: 'http://127.0.0.1:1234' }))
})

test('Config: verbose defaults to false', () => {
  const config = Config({ upstreamOrigin: 'http://127.0.0.1:1234', listenPort: 0 })
  assert.equal(config.verbose, false)
})

test('apply(): logs nothing by default, even startup', async () => {
  const originalLog = console.log
  const logged = []
  console.log = (message) => logged.push(message)
  const ctx = fakeCtx()
  try {
    const config = Config({ upstreamOrigin: 'http://127.0.0.1:1', listenPort: 0 })
    apply(ctx, config)
    await new Promise(resolve => setImmediate(resolve))
    assert.equal(logged.length, 0)
  } finally {
    console.log = originalLog
    ctx.stop()
  }
})

test('apply(): verbose:true logs the startup banner', async () => {
  const originalLog = console.log
  const logged = []
  console.log = (message) => logged.push(message)
  const ctx = fakeCtx()
  try {
    const config = Config({ upstreamOrigin: 'http://127.0.0.1:1', listenPort: 0, verbose: true })
    apply(ctx, config)
    await new Promise(resolve => setImmediate(resolve))
    assert.equal(logged.length, 1)
    assert.match(logged[0], /proxying/)
  } finally {
    console.log = originalLog
    ctx.stop()
  }
})
