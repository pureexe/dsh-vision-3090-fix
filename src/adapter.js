/**
 * LLM adapter for a self-hosted OpenAI-compatible `chat/completions` backend
 * (e.g. vLLM) that enforces "at most 1 image per prompt". Capping the image
 * count is the one behavior this adapter adds over a plain OpenAI-compatible
 * client; everything else is a minimal, direct implementation of the wire
 * protocol.
 *
 * @module dsh-vision-3090-fix/adapter
 */

import { LlmAdapter, LlmError, attributionHeaders } from '@deepseek-ai/dsh-llm'
import { buildWireMessages } from './wire.js'
import { translate } from './translate.js'

/** One provider HTTP request's fixed budget for resolving image bytes. */
function imagePolicyOf(config) {
  return { maxPixels: config.requestImagePixelBudget, maxBytes: config.requestImageMaxBytes }
}

export class Vision3090Adapter extends LlmAdapter {
  /**
   * @param attachments - `ctx.attachments` (or a compatible test double) resolving durable image references to request bytes.
   * @param config - validated plugin `Config`.
   */
  constructor(attachments, config) {
    super()
    this.attachments = attachments
    this.config = config
    this.models = new Map((config.models ?? []).map(model => [model.id, model]))
  }

  /** @returns metadata for one exact model id, falling back to the route's defaults for an undescribed model. */
  async resolveModel(provider, model) {
    const entry = this.models.get(model)
    const inputModalities = entry?.input ?? ['text']
    const reasoning = entry?.reasoningEfforts && Object.keys(entry.reasoningEfforts).length > 0
      ? {
        efforts: Object.keys(entry.reasoningEfforts).map(id => ({ id, name: id })),
        ...entry.defaultReasoningEffort === undefined ? {} : { defaultEffort: entry.defaultReasoningEffort },
      }
      : undefined
    return {
      provider,
      id: model,
      name: entry?.name ?? model,
      context: { contextWindow: entry?.contextWindow ?? this.config.defaultContextWindow },
      defaultMaxTokens: entry?.maxTokens ?? this.config.defaultMaxTokens,
      inputModalities,
      ...reasoning === undefined ? {} : { reasoning },
    }
  }

  /** @returns the statically configured model catalog for this route. */
  async listModels(provider) {
    return [...this.models.values()].map(model => ({
      provider,
      id: model.id,
      name: model.name ?? model.id,
      ...model.input === undefined ? {} : { inputModalities: model.input },
    }))
  }

  /** @returns the wire spelling configured for one opaque reasoning-effort id, or the id itself when undeclared. */
  wireReasoningEffort(model, effortId) {
    const entry = this.models.get(model)
    return entry?.reasoningEfforts?.[effortId] ?? effortId
  }

  /** @yields StreamChunk values for one `chat/completions` call, capping request images to `maxImagesPerRequest`. */
  async *stream(options) {
    const wireMessages = await buildWireMessages(options.messages, {
      attachments: this.attachments,
      imagePolicy: imagePolicyOf(this.config),
      maxImagesPerRequest: this.config.maxImagesPerRequest,
      signal: options.signal,
    })
    if (options.system !== undefined && options.system.length > 0) {
      wireMessages.unshift({ role: 'system', content: options.system })
    }

    const body = {
      model: options.model,
      messages: wireMessages,
      stream: true,
      stream_options: { include_usage: true },
      ...options.temperature === undefined ? {} : { temperature: options.temperature },
      ...options.maxTokens === undefined ? {} : { max_tokens: options.maxTokens },
      ...options.stop === undefined || options.stop.length === 0 ? {} : { stop: options.stop },
      ...options.tools === undefined || options.tools.length === 0 ? {} : {
        tools: options.tools.map(tool => ({
          type: 'function',
          function: { name: tool.name, description: tool.description, parameters: tool.parameters },
        })),
      },
      ...options.reasoningEffort === undefined ? {} : {
        reasoning_effort: this.wireReasoningEffort(options.model, options.reasoningEffort),
      },
    }

    const url = `${this.config.baseURL.replace(/\/+$/, '')}/chat/completions`
    let response
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${this.config.apiKey}`,
          ...attributionHeaders(),
        },
        body: JSON.stringify(body),
        ...options.signal ? { signal: options.signal } : {},
      })
    } catch (cause) {
      throw new LlmError(`pure-vision-fix: request to ${url} failed`, 'PROVIDER_HTTP_ERROR', { cause })
    }

    if (!response.ok || response.body === null) {
      const text = await response.text().catch(() => '')
      let message = `pure-vision-fix: provider HTTP ${response.status}`
      let code = 'PROVIDER_HTTP_ERROR'
      try {
        const parsed = JSON.parse(text)
        if (typeof parsed?.error?.message === 'string') message = parsed.error.message
        if (typeof parsed?.error?.code === 'string' && parsed.error.code.length > 0) code = parsed.error.code
      } catch {
        // Non-JSON error body: keep the default message and code.
      }
      throw new LlmError(message, code, { status: response.status })
    }

    yield* translate(response.body)
  }
}
