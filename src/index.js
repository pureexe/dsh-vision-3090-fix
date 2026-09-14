/**
 * dsh-vision-3090-fix: a DeepSeek Harness LLM adapter for self-hosted
 * OpenAI-compatible vision backends (e.g. a single-GPU vLLM server) that
 * reject any request carrying more than one image.
 *
 * Harness's shipped adapters offload images by accumulated byte size only
 * (`maxRequestImageBytes`), never by count, so a two-turn conversation with
 * two small images still sends both and the server answers:
 *
 *   "At most 1 image(s) may be provided in one prompt. (parameter=image)"
 *
 * This adapter keeps only the newest `maxImagesPerRequest` images (default 1)
 * and replaces every older one with the same stable placeholder text
 * `@deepseek-ai/dsh-llm`'s own byte-based offload uses, then speaks the
 * OpenAI-compatible `chat/completions` wire protocol directly.
 *
 * @module dsh-vision-3090-fix
 */

import Schema from '@deepseek-ai/schemastery'
import { Vision3090Adapter } from './adapter.js'

export const name = 'dsh-vision-3090-fix'
export const inject = ['llm', 'attachments']

const ModelConfig = Schema.object({
  id: Schema.string().required(),
  name: Schema.string(),
  contextWindow: Schema.number(),
  maxTokens: Schema.number(),
  input: Schema.array(Schema.string()).default(['text']),
  reasoningEfforts: Schema.dict(Schema.string()),
  defaultReasoningEffort: Schema.string(),
})

export const Config = Schema.object({
  /** Provider route names this adapter owns; must not collide with another registered adapter. */
  providers: Schema.array(Schema.string()).default(['pure']),
  /** Base URL of the OpenAI-compatible server, e.g. `http://10.204.100.243:1234/v1`. */
  baseURL: Schema.string().required(),
  /** Literal API key. Prefer `apiKeyEnv` so no secret enters cordis.yml/patch files. */
  apiKey: Schema.string(),
  /** Name of an environment variable read once at plugin load for the API key. */
  apiKeyEnv: Schema.string(),
  /** Images kept per request, newest first; every older image becomes placeholder text. */
  maxImagesPerRequest: Schema.number().min(1).default(1),
  /** Context-window fallback for a model id absent from `models`. */
  defaultContextWindow: Schema.number().default(131072),
  /** Output-token-cap fallback for a model id absent from `models`. */
  defaultMaxTokens: Schema.number().default(16384),
  /** Total-pixel budget applied when resolving one image's request bytes. */
  requestImagePixelBudget: Schema.number().default(4194304),
  /** Encoded-byte target applied when resolving one image's request bytes. */
  requestImageMaxBytes: Schema.number().default(1048576),
  /** Statically declared model catalog for this route. */
  models: Schema.array(ModelConfig).default([]),
})

/**
 * Mount the adapter and register it for every configured provider route.
 * @param ctx - Cordis context; requires the `llm` and `attachments` services.
 * @param config - validated {@link Config}.
 */
export function apply(ctx, config) {
  const apiKey = config.apiKey ?? (config.apiKeyEnv === undefined ? undefined : process.env[config.apiKeyEnv])
  if (apiKey === undefined || apiKey.length === 0) {
    throw new Error(
      'dsh-vision-3090-fix: no API key. Set config.apiKey, or config.apiKeyEnv naming an environment '
      + 'variable that is actually set (a keyless local server still needs a placeholder value).',
    )
  }
  if (config.maxImagesPerRequest < 1 || !Number.isInteger(config.maxImagesPerRequest)) {
    throw new Error('dsh-vision-3090-fix: maxImagesPerRequest must be a positive integer')
  }

  const adapter = new Vision3090Adapter(ctx.attachments, { ...config, apiKey })
  ctx.llm.registerAdapter(config.providers, adapter)
}
