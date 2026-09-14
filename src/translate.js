/**
 * Parse an OpenAI-compatible `chat/completions` SSE byte stream and translate
 * it into the Harness `StreamChunk` protocol.
 *
 * @module dsh-vision-3090-fix/translate
 */

import { EventSourceParserStream } from 'eventsource-parser/stream'
import { LlmError, EMPTY_RESPONSE_CODE } from '@deepseek-ai/dsh-llm'

const DONE = '[DONE]'

/** Parse an SSE byte stream into `data:` payloads, yielding `[DONE]` last. */
async function* parseSse(stream) {
  const events = stream
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new EventSourceParserStream())
  for await (const { data } of events) {
    yield data
    if (data === DONE) return
  }
  throw new LlmError('SSE stream ended without [DONE]', 'STREAM_CLOSED')
}

/** Map an OpenAI-compatible `finish_reason` string to a Harness FinishReason. */
function mapFinishReason(reason) {
  switch (reason) {
    case 'stop': return { kind: 'stop' }
    case 'tool_calls': return { kind: 'tool-calls' }
    case 'length': return { kind: 'max-tokens' }
    default:
      return { kind: 'error', failure: { message: `model stopped: ${reason}`, code: String(reason).toUpperCase() } }
  }
}

/** Map wire usage (vLLM's OpenAI-compatible shape) to disjoint Harness TokenUsage. */
function mapUsage(usage) {
  const cacheRead = usage.prompt_tokens_details?.cached_tokens
  const reasoning = usage.completion_tokens_details?.reasoning_tokens
  const combined = usage.prompt_tokens + usage.completion_tokens
  return {
    inputTokens: usage.prompt_tokens - (cacheRead ?? 0),
    outputTokens: usage.completion_tokens,
    ...usage.total_tokens === undefined ? { totalTokens: combined } : { totalTokens: usage.total_tokens },
    ...cacheRead === undefined ? {} : { cacheReadTokens: cacheRead },
    ...reasoning === undefined ? {} : { reasoningTokens: reasoning },
  }
}

/** A streamed tool-call identity is sent once; a later empty/null delta means "no update". */
function acceptIdentity(current, incoming) {
  return typeof incoming === 'string' && incoming.length > 0 ? incoming : current
}

function closeBlock(block) {
  switch (block.kind) {
    case 'text': return { type: 'text', text: block.text }
    case 'reasoning': return { type: 'reasoning', text: block.text }
    case 'tool-call':
      return { type: 'tool-call', id: block.callId ?? '', name: block.name ?? '', arguments: block.text }
    default: return { type: 'text', text: block.text }
  }
}

/**
 * Translate SSE `data:` payloads (`[DONE]`-terminated) from an OpenAI-compatible
 * `chat/completions` stream into Harness `StreamChunk` values.
 * @param stream - the raw response body of a `stream: true` request.
 * @returns block-start/delta/block-end chunks, then usage, then one finish chunk.
 */
export async function* translate(stream) {
  let nextIndex = 0
  let textBlock
  let reasoningBlock
  const toolBlocks = new Map()
  const order = []
  let pendingFinish
  let pendingUsage

  function open(kind) {
    const block = { index: nextIndex++, kind, text: '' }
    order.push(block)
    return block
  }

  for await (const payload of parseSse(stream)) {
    if (payload === DONE) {
      for (const block of order) {
        yield { type: 'block-end', index: block.index, block: closeBlock(block) }
      }
      if (pendingUsage) yield { type: 'usage', usage: pendingUsage }
      const reason = pendingFinish ?? { kind: 'stop' }
      yield {
        type: 'finish',
        reason: reason.kind === 'stop' && order.length === 0
          ? { kind: 'error', failure: { message: 'model returned a completed response with no content', code: EMPTY_RESPONSE_CODE } }
          : reason,
      }
      return
    }

    let chunk
    try {
      chunk = JSON.parse(payload)
    } catch {
      throw new LlmError(`malformed SSE payload: ${payload.slice(0, 120)}`, 'MALFORMED_RESPONSE')
    }

    for (const choice of chunk.choices ?? []) {
      const delta = choice.delta

      const reasoning = delta?.reasoning_content
      if (typeof reasoning === 'string' && reasoning.length > 0) {
        if (!reasoningBlock) {
          reasoningBlock = open('reasoning')
          yield { type: 'block-start', index: reasoningBlock.index, blockType: 'reasoning' }
        }
        reasoningBlock.text += reasoning
        yield { type: 'reasoning-delta', index: reasoningBlock.index, text: reasoning }
      }

      const content = delta?.content
      if (typeof content === 'string' && content.length > 0) {
        if (!textBlock) {
          textBlock = open('text')
          yield { type: 'block-start', index: textBlock.index, blockType: 'text' }
        }
        textBlock.text += content
        yield { type: 'text-delta', index: textBlock.index, text: content }
      }

      for (const call of delta?.tool_calls ?? []) {
        let block = toolBlocks.get(call.index)
        if (!block) {
          block = open('tool-call')
          toolBlocks.set(call.index, block)
          yield { type: 'block-start', index: block.index, blockType: 'tool-call' }
        }
        block.callId = acceptIdentity(block.callId, call.id)
        block.name = acceptIdentity(block.name, call.function?.name)
        const fragment = call.function?.arguments ?? ''
        block.text += fragment
        yield {
          type: 'tool-call-delta',
          index: block.index,
          id: block.callId ?? '',
          ...block.name === undefined ? {} : { name: block.name },
          argumentsDelta: fragment,
        }
      }

      if (typeof choice.finish_reason === 'string') pendingFinish = mapFinishReason(choice.finish_reason)
    }

    if (chunk.usage) pendingUsage = mapUsage(chunk.usage)
  }

  throw new LlmError('SSE payload stream ended without [DONE]', 'STREAM_CLOSED')
}
