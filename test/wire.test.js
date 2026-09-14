import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createUserMessage, createAssistantMessage, createToolResultMessage } from '@deepseek-ai/dsh-llm'
import { buildWireMessages } from '../src/wire.js'

/** Minimal `ctx.attachments`-shaped double: returns deterministic fake bytes for any ref. */
function fakeAttachments() {
  const calls = []
  return {
    calls,
    async readImageRequest(ref) {
      calls.push(ref.attachmentId)
      return {
        data: new TextEncoder().encode(`bytes-of-${ref.attachmentId}`),
        mediaType: ref.mediaType,
        bytes: 10,
        width: ref.width,
        height: ref.height,
      }
    },
  }
}

function imageRef(id) {
  return { attachmentId: id, mediaType: 'image/png', bytes: 100, width: 8, height: 8 }
}

test('a single image in one request is kept and base64-encoded', async () => {
  const message = createUserMessage({
    content: [
      { type: 'text', text: 'What is this?' },
      { type: 'image', attachment: imageRef('sha256:one') },
    ],
    source: { kind: 'user' },
  })
  const attachments = fakeAttachments()
  const wire = await buildWireMessages([message], {
    attachments,
    imagePolicy: { maxPixels: 4194304, maxBytes: 1048576 },
    maxImagesPerRequest: 1,
  })

  assert.equal(wire.length, 1)
  assert.equal(wire[0].role, 'user')
  const imageParts = wire[0].content.filter(part => part.type === 'image_url')
  assert.equal(imageParts.length, 1)
  assert.match(imageParts[0].image_url.url, /^data:image\/png;base64,/)
  assert.deepEqual(attachments.calls, ['sha256:one'])
})

test('two images across two turns: only the newest is sent, matching a 1-image-per-prompt backend', async () => {
  const first = createUserMessage({
    content: [
      { type: 'text', text: 'First image.' },
      { type: 'image', attachment: imageRef('sha256:first') },
    ],
    source: { kind: 'user' },
  })
  const reply = createAssistantMessage({
    content: [{ type: 'text', text: 'I see the first image.' }],
    source: { provider: 'pure', model: 'qwen3.8-27b' },
  })
  const second = createUserMessage({
    content: [
      { type: 'text', text: 'Second image, what changed?' },
      { type: 'image', attachment: imageRef('sha256:second') },
    ],
    source: { kind: 'user' },
  })

  const attachments = fakeAttachments()
  const wire = await buildWireMessages([first, reply, second], {
    attachments,
    imagePolicy: { maxPixels: 4194304, maxBytes: 1048576 },
    maxImagesPerRequest: 1,
  })

  // The whole point of this adapter: the backend can only take one image, so
  // exactly one image_url part reaches the wire request no matter how many
  // images the conversation has accumulated.
  const allImageParts = wire.flatMap(m => Array.isArray(m.content) ? m.content.filter(p => p.type === 'image_url') : [])
  assert.equal(allImageParts.length, 1)

  // The offloaded (older) image is never even read for bytes.
  assert.deepEqual(attachments.calls, ['sha256:second'])

  // The older image's message still carries readable placeholder text instead
  // of silently vanishing.
  const firstWireUser = wire.find(m => m.role === 'user' && typeof m.content === 'string')
  assert.match(firstWireUser.content, /image omitted/)
})

test('maxImagesPerRequest can keep more than one image', async () => {
  const first = createUserMessage({
    content: [{ type: 'image', attachment: imageRef('sha256:a') }],
    source: { kind: 'user' },
  })
  const second = createUserMessage({
    content: [{ type: 'image', attachment: imageRef('sha256:b') }],
    source: { kind: 'user' },
  })
  const attachments = fakeAttachments()
  const wire = await buildWireMessages([first, second], {
    attachments,
    imagePolicy: { maxPixels: 4194304, maxBytes: 1048576 },
    maxImagesPerRequest: 2,
  })
  const allImageParts = wire.flatMap(m => Array.isArray(m.content) ? m.content.filter(p => p.type === 'image_url') : [])
  assert.equal(allImageParts.length, 2)
})

test('tool-result messages become their own `tool` wire message', async () => {
  const result = createToolResultMessage({
    callId: 'call-1',
    content: [{ type: 'text', text: 'file contents' }],
    isError: false,
  })
  const attachments = fakeAttachments()
  const wire = await buildWireMessages([result], {
    attachments,
    imagePolicy: { maxPixels: 4194304, maxBytes: 1048576 },
    maxImagesPerRequest: 1,
  })
  assert.equal(wire.length, 1)
  assert.equal(wire[0].role, 'tool')
  assert.equal(wire[0].tool_call_id, 'call-1')
  assert.equal(wire[0].content, 'file contents')
})

test('system-role history text is forwarded as a system wire message', async () => {
  const system = { role: 'system', content: [{ type: 'text', text: 'be nice' }], id: 's1', source: { kind: 'plugin', plugin: 'x' } }
  const user = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } })
  const attachments = fakeAttachments()
  const wire = await buildWireMessages([system, user], {
    attachments,
    imagePolicy: { maxPixels: 4194304, maxBytes: 1048576 },
    maxImagesPerRequest: 1,
  })
  assert.deepEqual(wire[0], { role: 'system', content: 'be nice' })
})
