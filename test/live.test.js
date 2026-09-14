/**
 * End-to-end test against the real backend from the bug report, through an
 * actual instance of this proxy. Skipped unless VISION_3090_FIX_LIVE_UPSTREAM
 * is set, since it needs network access to a specific vLLM server and a real
 * API key.
 *
 * Run with:
 *   VISION_3090_FIX_LIVE_UPSTREAM=http://10.204.100.243:1234 \
 *   VISION_3090_FIX_LIVE_API_KEY=<your-api-key> \
 *   VISION_3090_FIX_LIVE_MODEL=qwen3.8-27b \
 *   VISION_3090_FIX_LIVE_IMAGE=/home/pakkapon/a.png \
 *   node --test test/live.test.js
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import { once } from 'node:events'
import { createProxyServer } from '../src/proxy.js'

const upstreamOrigin = process.env.VISION_3090_FIX_LIVE_UPSTREAM
const apiKey = process.env.VISION_3090_FIX_LIVE_API_KEY
const model = process.env.VISION_3090_FIX_LIVE_MODEL ?? 'qwen3.8-27b'
const imagePath = process.env.VISION_3090_FIX_LIVE_IMAGE ?? '/home/pakkapon/a.png'

const shouldRun = upstreamOrigin !== undefined && apiKey !== undefined

async function startProxy(maxImagesPerRequest) {
  const server = createProxyServer({ upstreamOrigin, maxImagesPerRequest })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const { port } = server.address()
  return { server, baseURL: `http://127.0.0.1:${port}/v1` }
}

async function chatWithImages(baseURL, imageCount) {
  const bytes = await fs.readFile(imagePath)
  const b64 = bytes.toString('base64')
  const image = { type: 'image_url', image_url: { url: `data:image/png;base64,${b64}` } }
  const messages = []
  for (let i = 0; i < imageCount; i += 1) {
    messages.push({ role: 'user', content: [{ type: 'text', text: `image ${i + 1}` }, image] })
    if (i < imageCount - 1) messages.push({ role: 'assistant', content: 'ok' })
  }
  const response = await fetch(`${baseURL}/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model, messages, max_tokens: 64 }),
  })
  const body = await response.json()
  return { status: response.status, body }
}

test('without the proxy fix (cap effectively unbounded), two images reproduces the reported 400', { skip: !shouldRun }, async () => {
  const { server, baseURL } = await startProxy(Number.MAX_SAFE_INTEGER)
  try {
    const { status, body } = await chatWithImages(baseURL, 2)
    assert.equal(status, 400)
    assert.match(body.error.message, /At most 1 image/)
  } finally {
    server.close()
  }
})

test('the fix: proxying through a 1-image cap lets the same two-image conversation succeed', { skip: !shouldRun }, async () => {
  const { server, baseURL } = await startProxy(1)
  try {
    const { status, body } = await chatWithImages(baseURL, 2)
    assert.equal(status, 200)
    assert.ok(body.choices?.[0])
  } finally {
    server.close()
  }
})

test('a single image passes through the proxy untouched and still works', { skip: !shouldRun }, async () => {
  const { server, baseURL } = await startProxy(1)
  try {
    const { status, body } = await chatWithImages(baseURL, 1)
    assert.equal(status, 200)
    assert.ok(body.choices?.[0])
  } finally {
    server.close()
  }
})
