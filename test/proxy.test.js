import { test } from 'node:test'
import assert from 'node:assert/strict'
import http from 'node:http'
import { once } from 'node:events'
import { capImagesInMessages, createProxyServer, OMITTED_IMAGE_TEXT } from '../src/proxy.js'

function imagePart(id) {
  return { type: 'image_url', image_url: { url: `data:image/png;base64,${id}` } }
}

test('capImagesInMessages: fewer images than the cap is left untouched (same reference)', () => {
  const messages = [{ role: 'user', content: [{ type: 'text', text: 'hi' }, imagePart('a')] }]
  const result = capImagesInMessages(messages, 1)
  assert.equal(result, messages)
})

test('capImagesInMessages: two images across two turns keeps only the newest', () => {
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'first' }, imagePart('a')] },
    { role: 'assistant', content: 'ok' },
    { role: 'user', content: [{ type: 'text', text: 'second' }, imagePart('b')] },
  ]
  const result = capImagesInMessages(messages, 1)

  const allImageParts = result.flatMap(m => (Array.isArray(m.content) ? m.content.filter(p => p.type === 'image_url') : []))
  assert.equal(allImageParts.length, 1)
  assert.equal(allImageParts[0].image_url.url, 'data:image/png;base64,b')

  const firstMessage = result[0]
  const placeholder = firstMessage.content.find(p => p.type === 'text' && p.text === OMITTED_IMAGE_TEXT)
  assert.ok(placeholder, 'the older image should become placeholder text in place')
  // Untouched messages are not cloned.
  assert.equal(result[1], messages[1])
})

test('capImagesInMessages: images returned by tool calls are capped the same as user images', () => {
  const messages = [
    { role: 'tool', tool_call_id: 'call-1', content: [imagePart('screenshot')] },
    { role: 'tool', tool_call_id: 'call-2', content: [imagePart('pulled-file')] },
  ]
  const result = capImagesInMessages(messages, 1)
  const kept = result.flatMap(m => m.content.filter(p => p.type === 'image_url'))
  assert.equal(kept.length, 1)
  assert.equal(kept[0].image_url.url, 'data:image/png;base64,pulled-file')
})

test('capImagesInMessages: maxImages can keep more than one', () => {
  const messages = [
    { role: 'user', content: [imagePart('a')] },
    { role: 'user', content: [imagePart('b')] },
  ]
  const result = capImagesInMessages(messages, 2)
  assert.equal(result, messages)
})

test('capImagesInMessages: non-array messages/content pass through unchanged', () => {
  assert.equal(capImagesInMessages(undefined, 1), undefined)
  const messages = [{ role: 'system', content: 'be nice' }]
  assert.equal(capImagesInMessages(messages, 1), messages)
})

/** Start a fake upstream OpenAI-compatible server recording what it received. */
async function startFakeUpstream() {
  const received = []
  const server = http.createServer(async (req, res) => {
    const chunks = []
    for await (const chunk of req) chunks.push(chunk)
    const body = Buffer.concat(chunks).toString('utf8')
    received.push({ method: req.method, url: req.url, headers: req.headers, body })
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ ok: true, echoedImageCount: (JSON.parse(body || '{}').messages ?? []).flatMap(m => Array.isArray(m.content) ? m.content.filter(p => p.type === 'image_url') : []).length }))
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const { port } = server.address()
  received.origin = `http://127.0.0.1:${port}`
  return { server, received }
}

test('proxy end-to-end: forwards a capped request and streams the upstream response back', async () => {
  const { server: upstream, received } = await startFakeUpstream()
  const proxy = createProxyServer({ upstreamOrigin: received.origin, maxImagesPerRequest: 1 })
  proxy.listen(0, '127.0.0.1')
  await once(proxy, 'listening')
  const { port } = proxy.address()

  const requestBody = JSON.stringify({
    model: 'qwen3.8-27b',
    messages: [
      { role: 'user', content: [{ type: 'text', text: 'first' }, imagePart('a')] },
      { role: 'user', content: [{ type: 'text', text: 'second' }, imagePart('b')] },
    ],
  })

  const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer test-key' },
    body: requestBody,
  })
  const json = await response.json()

  assert.equal(response.status, 200)
  assert.equal(json.echoedImageCount, 1)
  assert.equal(received.length, 1)
  assert.equal(received[0].headers.authorization, 'Bearer test-key')
  const forwarded = JSON.parse(received[0].body)
  const forwardedImages = forwarded.messages.flatMap(m => m.content.filter(p => p.type === 'image_url'))
  assert.equal(forwardedImages.length, 1)

  proxy.close()
  upstream.close()
})

test('proxy end-to-end: a request already within the cap is forwarded unmodified', async () => {
  const { server: upstream, received } = await startFakeUpstream()
  const proxy = createProxyServer({ upstreamOrigin: received.origin, maxImagesPerRequest: 1 })
  proxy.listen(0, '127.0.0.1')
  await once(proxy, 'listening')
  const { port } = proxy.address()

  const requestBody = JSON.stringify({
    model: 'qwen3.8-27b',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }, imagePart('a')] }],
  })
  const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: requestBody,
  })
  await response.json()
  assert.equal(JSON.parse(received[0].body).messages[0].content.length, 2)

  proxy.close()
  upstream.close()
})

test('proxy end-to-end: non-JSON requests (e.g. GET /models) are passed through untouched', async () => {
  const { server: upstream, received } = await startFakeUpstream()
  const proxy = createProxyServer({ upstreamOrigin: received.origin, maxImagesPerRequest: 1 })
  proxy.listen(0, '127.0.0.1')
  await once(proxy, 'listening')
  const { port } = proxy.address()

  const response = await fetch(`http://127.0.0.1:${port}/v1/models`)
  await response.json()
  assert.equal(received[0].method, 'GET')
  assert.equal(received[0].url, '/v1/models')

  proxy.close()
  upstream.close()
})

test('proxy end-to-end: models config scopes the cap to specific model ids', async () => {
  const { server: upstream, received } = await startFakeUpstream()
  const proxy = createProxyServer({
    upstreamOrigin: received.origin,
    maxImagesPerRequest: 1,
    models: ['qwen3.8-27b'],
  })
  proxy.listen(0, '127.0.0.1')
  await once(proxy, 'listening')
  const { port } = proxy.address()

  const twoImages = (model) => JSON.stringify({
    model,
    messages: [
      { role: 'user', content: [{ type: 'text', text: 'first' }, imagePart('a')] },
      { role: 'user', content: [{ type: 'text', text: 'second' }, imagePart('b')] },
    ],
  })

  // In scope: capped down to 1 image.
  await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: twoImages('qwen3.8-27b'),
  })
  const cappedRequest = JSON.parse(received[0].body)
  const cappedImages = cappedRequest.messages.flatMap(m => m.content.filter(p => p.type === 'image_url'))
  assert.equal(cappedImages.length, 1)

  // Out of scope: forwarded with both images, completely untouched.
  await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: twoImages('meta/muse-glimmer'),
  })
  assert.equal(received[1].body, twoImages('meta/muse-glimmer'))
  const untouchedRequest = JSON.parse(received[1].body)
  const untouchedImages = untouchedRequest.messages.flatMap(m => m.content.filter(p => p.type === 'image_url'))
  assert.equal(untouchedImages.length, 2)

  proxy.close()
  upstream.close()
})

test('proxy end-to-end: no models config caps every model (default, backward compatible)', async () => {
  const { server: upstream, received } = await startFakeUpstream()
  const proxy = createProxyServer({ upstreamOrigin: received.origin, maxImagesPerRequest: 1 })
  proxy.listen(0, '127.0.0.1')
  await once(proxy, 'listening')
  const { port } = proxy.address()

  await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: 'meta/muse-glimmer',
      messages: [
        { role: 'user', content: [imagePart('a')] },
        { role: 'user', content: [imagePart('b')] },
      ],
    }),
  })
  const forwarded = JSON.parse(received[0].body)
  const images = forwarded.messages.flatMap(m => m.content.filter(p => p.type === 'image_url'))
  assert.equal(images.length, 1)

  proxy.close()
  upstream.close()
})
