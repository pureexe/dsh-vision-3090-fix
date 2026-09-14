import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Config, apply } from '../src/index.js'

function fakeCtx() {
  const registered = []
  return {
    attachments: { async readImageRequest() { throw new Error('not used in this test') } },
    llm: {
      registerAdapter(providers, adapter) {
        registered.push({ providers, adapter })
        return () => {}
      },
    },
    registered,
  }
}

test('Config fills in documented defaults', () => {
  const config = Config({ baseURL: 'http://127.0.0.1:1234/v1', apiKey: 'k' })
  assert.equal(config.maxImagesPerRequest, 1)
  assert.deepEqual(config.providers, ['pure'])
  assert.equal(config.defaultContextWindow, 131072)
  assert.equal(config.requestImagePixelBudget, 4194304)
})

test('Config requires baseURL', () => {
  assert.throws(() => Config({ apiKey: 'k' }))
})

test('apply() registers an adapter for every configured provider route', () => {
  const ctx = fakeCtx()
  const config = Config({ baseURL: 'http://127.0.0.1:1234/v1', apiKey: 'k', providers: ['pure', 'pure-2'] })
  apply(ctx, config)
  assert.equal(ctx.registered.length, 1)
  assert.deepEqual(ctx.registered[0].providers, ['pure', 'pure-2'])
})

test('apply() reads the API key from apiKeyEnv when apiKey is not set directly', () => {
  process.env.VISION_3090_FIX_TEST_KEY = 'from-env'
  const ctx = fakeCtx()
  const config = Config({ baseURL: 'http://127.0.0.1:1234/v1', apiKeyEnv: 'VISION_3090_FIX_TEST_KEY' })
  apply(ctx, config)
  assert.equal(ctx.registered[0].adapter.config.apiKey, 'from-env')
  delete process.env.VISION_3090_FIX_TEST_KEY
})

test('apply() fails loudly when no API key is available', () => {
  const ctx = fakeCtx()
  const config = Config({ baseURL: 'http://127.0.0.1:1234/v1', apiKeyEnv: 'VISION_3090_FIX_MISSING_KEY' })
  assert.throws(() => apply(ctx, config), /no API key/)
})
