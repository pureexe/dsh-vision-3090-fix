/**
 * End-to-end test against the real backend from the bug report. Skipped
 * unless VISION_3090_FIX_LIVE_BASE_URL is set, since it needs network access
 * to a specific vLLM server and a real API key.
 *
 * Run with:
 *   VISION_3090_FIX_LIVE_BASE_URL=http://10.204.100.243:1234/v1 \
 *   VISION_3090_FIX_LIVE_API_KEY=<your-api-key> \
 *   VISION_3090_FIX_LIVE_MODEL=qwen3.8-27b \
 *   VISION_3090_FIX_LIVE_IMAGE=/home/pakkapon/a.png \
 *   node --test test/live.test.js
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import { createUserMessage, createAssistantMessage, createToolResultMessage } from '@deepseek-ai/dsh-llm'
import { Vision3090Adapter } from '../src/adapter.js'

const baseURL = process.env.VISION_3090_FIX_LIVE_BASE_URL
const apiKey = process.env.VISION_3090_FIX_LIVE_API_KEY
const model = process.env.VISION_3090_FIX_LIVE_MODEL ?? 'qwen3.8-27b'
const imagePath = process.env.VISION_3090_FIX_LIVE_IMAGE ?? '/home/pakkapon/a.png'

const shouldRun = baseURL !== undefined && apiKey !== undefined

test('reproduces the reported bug directly against the real vLLM server', { skip: !shouldRun }, async () => {
  const bytes = await fs.readFile(imagePath)
  const attachments = {
    async readImageRequest(ref) {
      return { data: bytes, mediaType: 'image/png', bytes: bytes.length, width: 1, height: 1 }
    },
  }

  // No cap at all (maxImagesPerRequest effectively unbounded): this is what
  // ships today and is exactly what produced the reported 400.
  const uncapped = new Vision3090Adapter(attachments, {
    baseURL,
    apiKey,
    maxImagesPerRequest: Number.MAX_SAFE_INTEGER,
    defaultContextWindow: 131072,
    defaultMaxTokens: 64,
    requestImagePixelBudget: 4194304,
    requestImageMaxBytes: 1048576,
    models: [],
  })

  const first = createUserMessage({
    content: [{ type: 'text', text: 'First image.' }, { type: 'image', attachment: { attachmentId: 'sha256:a', mediaType: 'image/png', bytes: bytes.length, width: 1, height: 1 } }],
    source: { kind: 'user' },
  })
  const reply = createAssistantMessage({
    content: [{ type: 'text', text: 'ok' }],
    source: { provider: 'pure', model },
  })
  const second = createUserMessage({
    content: [{ type: 'text', text: 'Second image, what changed?' }, { type: 'image', attachment: { attachmentId: 'sha256:b', mediaType: 'image/png', bytes: bytes.length, width: 1, height: 1 } }],
    source: { kind: 'user' },
  })

  await assert.rejects(async () => {
    for await (const _chunk of uncapped.stream({ provider: 'pure', model, messages: [first, reply, second], maxTokens: 16 })) {
      // draining the stream is enough to surface the thrown LlmError
    }
  }, (error) => {
    assert.match(error.message, /At most 1 image/)
    return true
  })
})

test('the fix: capping to 1 image lets the same two-image conversation succeed', { skip: !shouldRun }, async () => {
  const bytes = await fs.readFile(imagePath)
  const attachments = {
    async readImageRequest() {
      return { data: bytes, mediaType: 'image/png', bytes: bytes.length, width: 1, height: 1 }
    },
  }

  const fixed = new Vision3090Adapter(attachments, {
    baseURL,
    apiKey,
    maxImagesPerRequest: 1,
    defaultContextWindow: 131072,
    defaultMaxTokens: 64,
    requestImagePixelBudget: 4194304,
    requestImageMaxBytes: 1048576,
    models: [],
  })

  const first = createUserMessage({
    content: [{ type: 'text', text: 'First image.' }, { type: 'image', attachment: { attachmentId: 'sha256:a', mediaType: 'image/png', bytes: bytes.length, width: 1, height: 1 } }],
    source: { kind: 'user' },
  })
  const reply = createAssistantMessage({
    content: [{ type: 'text', text: 'ok' }],
    source: { provider: 'pure', model },
  })
  const second = createUserMessage({
    content: [{ type: 'text', text: 'Second image, what changed?' }, { type: 'image', attachment: { attachmentId: 'sha256:b', mediaType: 'image/png', bytes: bytes.length, width: 1, height: 1 } }],
    source: { kind: 'user' },
  })

  const chunks = []
  for await (const chunk of fixed.stream({ provider: 'pure', model, messages: [first, reply, second], maxTokens: 64 })) {
    chunks.push(chunk)
  }

  const finish = chunks.at(-1)
  assert.equal(finish.type, 'finish')
  // A reasoning model may spend the whole budget thinking before any visible
  // text ('max-tokens'), or answer and stop normally ('stop'); either is a
  // real 200 OK response, proving the request no longer gets the reported
  // "At most 1 image(s)" 400. What must never happen is 'error'.
  assert.notEqual(finish.reason.kind, 'error')
  assert.ok(chunks.some(c => c.type === 'usage'))
})

test('two tool calls in one turn each returning an image succeeds against the real backend', { skip: !shouldRun }, async () => {
  const bytes = await fs.readFile(imagePath)
  const attachments = {
    async readImageRequest() {
      return { data: bytes, mediaType: 'image/png', bytes: bytes.length, width: 1, height: 1 }
    },
  }
  const fixed = new Vision3090Adapter(attachments, {
    baseURL,
    apiKey,
    maxImagesPerRequest: 1,
    defaultContextWindow: 131072,
    defaultMaxTokens: 64,
    requestImagePixelBudget: 4194304,
    requestImageMaxBytes: 1048576,
    models: [],
  })

  const priorCall = createAssistantMessage({
    content: [
      { type: 'tool-call', id: 'call-screenshot', name: 'screen_shot', arguments: '{}' },
      { type: 'tool-call', id: 'call-pull', name: 'filesystem_pull', arguments: '{}' },
    ],
    source: { provider: 'pure', model },
  })
  const screenshot = createToolResultMessage({
    callId: 'call-screenshot',
    content: [{ type: 'image', attachment: { attachmentId: 'sha256:screenshot', mediaType: 'image/png', bytes: bytes.length, width: 1, height: 1 } }],
    isError: false,
  })
  const pulledFile = createToolResultMessage({
    callId: 'call-pull',
    content: [{ type: 'image', attachment: { attachmentId: 'sha256:pulled', mediaType: 'image/png', bytes: bytes.length, width: 1, height: 1 } }],
    isError: false,
  })
  const ask = createUserMessage({ content: [{ type: 'text', text: 'What do you see in the most recent image?' }], source: { kind: 'user' } })

  const chunks = []
  for await (const chunk of fixed.stream({
    provider: 'pure',
    model,
    messages: [priorCall, screenshot, pulledFile, ask],
    maxTokens: 64,
  })) {
    chunks.push(chunk)
  }
  const finish = chunks.at(-1)
  assert.equal(finish.type, 'finish')
  assert.notEqual(finish.reason.kind, 'error')
  assert.ok(chunks.some(c => c.type === 'usage'))
})
