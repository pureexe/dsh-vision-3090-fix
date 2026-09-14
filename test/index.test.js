import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Config } from '../src/index.js'

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
