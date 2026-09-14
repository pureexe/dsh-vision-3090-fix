/**
 * dsh-vision-3090-fix: a local reverse proxy for a self-hosted
 * OpenAI-compatible vision backend (e.g. a single-GPU vLLM server) that
 * rejects any request carrying more than one image.
 *
 * Harness's shipped adapters (dsh-llm-pi-ai, dsh-llm-deepseek) offload
 * request images by accumulated byte size only, never by count, so a
 * two-turn conversation with two small images still sends both and the
 * server answers:
 *
 *   "At most 1 image(s) may be provided in one prompt. (parameter=image)"
 *
 * Rather than replacing your existing LLM adapter/provider config, this
 * plugin starts a small local HTTP proxy that keeps only the newest
 * `maxImagesPerRequest` images in each forwarded `chat/completions` request.
 * Point your provider's `baseURL` (in settings.yaml or cordis.yml, however
 * you already configure it) at this proxy instead of at the backend
 * directly — everything else (credentials, model list, streaming) stays
 * exactly as it was.
 *
 * @module dsh-vision-3090-fix
 */

import Schema from '@deepseek-ai/schemastery'
import { createProxyServer } from './proxy.js'

export const name = 'dsh-vision-3090-fix'

export const Config = Schema.object({
  /** Scheme+host+port of the real backend, e.g. `http://10.204.100.243:1234` (no path). */
  upstreamOrigin: Schema.string().required(),
  /** Host the proxy listens on. */
  listenHost: Schema.string().default('127.0.0.1'),
  /** Port the proxy listens on. Point your provider's `baseURL` at this host:port. */
  listenPort: Schema.number().required(),
  /** Images kept per forwarded request, newest first; every older image becomes placeholder text. */
  maxImagesPerRequest: Schema.number().min(1).default(1),
  /** Model ids the cap applies to; every other model is forwarded completely untouched. Empty means every model. */
  models: Schema.array(Schema.string()).default([]),
})

/**
 * Start the proxy and stop it cleanly when the plugin unloads.
 * @param ctx - Cordis context.
 * @param config - validated {@link Config}.
 */
export function apply(ctx, config) {
  if (!Number.isInteger(config.maxImagesPerRequest) || config.maxImagesPerRequest < 1) {
    throw new Error('dsh-vision-3090-fix: maxImagesPerRequest must be a positive integer')
  }

  const server = createProxyServer({
    upstreamOrigin: config.upstreamOrigin,
    maxImagesPerRequest: config.maxImagesPerRequest,
    models: config.models,
    log: message => console.log(`[dsh-vision-3090-fix] ${message}`),
  })

  ctx.effect(() => {
    server.listen(config.listenPort, config.listenHost)
    const url = `http://${config.listenHost}:${config.listenPort}`
    const scope = config.models.length > 0 ? `model(s) ${config.models.join(', ')}` : 'every model'
    console.log(`[dsh-vision-3090-fix] proxying ${url} -> ${config.upstreamOrigin}, capping ${scope} to ${config.maxImagesPerRequest} image(s) per request`)
    return () => {
      server.close()
    }
  })
}
