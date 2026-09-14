/**
 * Convert Harness provider-neutral messages into OpenAI-compatible
 * `chat/completions` wire messages, capping the number of images per request
 * to the count this backend's server actually accepts.
 *
 * @module dsh-vision-3090-fix/wire
 */

import { offloadRequestImagesWithPolicy, offloadedImageText } from '@deepseek-ai/dsh-llm'

/** Join text (and reasoning) blocks of one message into plain text. */
function flattenText(content) {
  return content
    .filter(block => block.type === 'text' || block.type === 'reasoning')
    .map(block => block.text)
    .join('')
}

/**
 * Convert a block list (a message's content, or one tool-result's nested
 * content) into OpenAI content: a plain string when it is text-only, or a
 * content-part array once any image survives the request-wide image cap.
 * Nested tool-result blocks are flattened in place, so an image returned by
 * a tool (a screenshot, a pulled file) is treated exactly like a directly
 * attached one: it counts toward, and can be kept by, `maxImagesPerRequest`.
 */
async function blocksToWireContent(blocks, attachments, imagePolicy, signal) {
  const parts = []
  for (const block of blocks) {
    if (block.type === 'text' || block.type === 'reasoning') {
      if (block.text.length > 0) parts.push({ type: 'text', text: block.text })
      continue
    }
    if (block.type === 'image') {
      const version = await attachments.readImageRequest(block.attachment, imagePolicy, signal)
      const base64 = Buffer.from(version.data).toString('base64')
      parts.push({ type: 'image_url', image_url: { url: `data:${version.mediaType};base64,${base64}` } })
      continue
    }
    if (block.type === 'tool-result') {
      const nested = await blocksToWireContent(block.content, attachments, imagePolicy, signal)
      if (typeof nested === 'string') {
        if (nested.length > 0) parts.push({ type: 'text', text: nested })
      } else {
        parts.push(...nested)
      }
      continue
    }
    // Other merge-extensible blocks are not wire vocabulary here.
  }
  if (parts.length === 0) return ''
  if (parts.every(part => part.type === 'text')) return parts.map(part => part.text).join('')
  return parts
}

/** Convert one assistant-role message into an OpenAI wire message. */
function assistantToWire(message) {
  const text = message.content.filter(block => block.type === 'text').map(block => block.text).join('')
  const toolCalls = message.content
    .filter(block => block.type === 'tool-call')
    .map(block => ({
      id: block.id,
      type: 'function',
      function: { name: block.name, arguments: block.arguments },
    }))
  const wire = { role: 'assistant', content: text.length > 0 ? text : null }
  if (toolCalls.length > 0) wire.tool_calls = toolCalls
  return wire
}

/**
 * Build the OpenAI-compatible `messages` array for one request, replacing
 * every image beyond `maxImagesPerRequest` (oldest first, tool-result images
 * included) with stable text so a single-image-per-prompt backend never
 * receives more than it accepts.
 * @param messages - Harness request history, oldest first.
 * @param options.attachments - `ctx.attachments`-shaped store resolving image bytes.
 * @param options.imagePolicy - `{ maxPixels, maxBytes }` passed to `readImageRequest`.
 * @param options.maxImagesPerRequest - images kept per request; excess is offloaded to text.
 * @param options.signal - forwarded to attachment resolution.
 * @returns OpenAI-compatible wire messages, oldest first.
 */
export async function buildWireMessages(messages, { attachments, imagePolicy, maxImagesPerRequest, signal }) {
  const capped = offloadRequestImagesWithPolicy(messages, {
    maxImages: maxImagesPerRequest,
    representation: 'base64',
    placeholder: ref => offloadedImageText(ref),
  })

  const wire = []
  for (const message of capped) {
    if (message.role === 'system') {
      const text = flattenText(message.content)
      if (text.length > 0) wire.push({ role: 'system', content: text })
      continue
    }
    if (message.role === 'assistant') {
      wire.push(assistantToWire(message))
      continue
    }
    // role === 'user': tool-result blocks become their own `tool` messages
    // (a returned image is kept, subject to the same cap above); everything
    // else becomes one `user` message.
    const toolResults = message.content.filter(block => block.type === 'tool-result')
    const regular = message.content.filter(block => block.type !== 'tool-result')
    if (regular.length > 0 || toolResults.length === 0) {
      const content = await blocksToWireContent(regular, attachments, imagePolicy, signal)
      wire.push({ role: 'user', content })
    }
    for (const result of toolResults) {
      const content = await blocksToWireContent(result.content, attachments, imagePolicy, signal)
      wire.push({
        role: 'tool',
        tool_call_id: result.toolCallId,
        content: content === '' ? '(no output)' : content,
      })
    }
  }
  return wire
}
