/**
 * A tiny local reverse proxy that sits in front of an OpenAI-compatible
 * `chat/completions` server and caps the number of images in each forwarded
 * request. This lets any client (llm-pi-ai, a raw curl, etc.) keep talking
 * plain OpenAI wire protocol to a normal `baseURL` — that URL just happens to
 * be this proxy instead of the real backend — with no changes to the
 * client's own configuration, credentials, or request/response handling.
 *
 * @module dsh-vision-3090-fix/proxy
 */

import http from 'node:http'
import { Readable } from 'node:stream'

/** Stable placeholder text for an image dropped by the count cap. */
export const OMITTED_IMAGE_TEXT = '[image omitted: capped to the newest image(s) this backend accepts]'

/**
 * Replace every `image_url` content part beyond the newest `maxImages`
 * (across the whole message list, in order) with a text placeholder.
 * Messages/content arrays are only cloned where an image is actually
 * replaced; everything else is returned by reference.
 * @param messages - an OpenAI-compatible wire `messages` array.
 * @param maxImages - images kept, newest first; must be a positive integer.
 * @returns the same array when nothing needs replacing, otherwise a shallow copy with excess images replaced.
 */
export function capImagesInMessages(messages, maxImages) {
  if (!Array.isArray(messages)) return messages

  const locations = []
  for (const [messageIndex, message] of messages.entries()) {
    if (!Array.isArray(message?.content)) continue
    for (const [partIndex, part] of message.content.entries()) {
      if (part?.type === 'image_url') locations.push({ messageIndex, partIndex })
    }
  }
  if (locations.length <= maxImages) return messages

  const toReplace = locations.slice(0, locations.length - maxImages)
  const replaceByMessage = new Map()
  for (const { messageIndex, partIndex } of toReplace) {
    if (!replaceByMessage.has(messageIndex)) replaceByMessage.set(messageIndex, new Set())
    replaceByMessage.get(messageIndex).add(partIndex)
  }

  return messages.map((message, messageIndex) => {
    const partIndexes = replaceByMessage.get(messageIndex)
    if (partIndexes === undefined) return message
    const content = message.content.map((part, partIndex) => (
      partIndexes.has(partIndex) ? { type: 'text', text: OMITTED_IMAGE_TEXT } : part
    ))
    return { ...message, content }
  })
}

/** Read a Node request body fully into a Buffer. */
async function readBody(req) {
  const chunks = []
  for await (const chunk of req) chunks.push(chunk)
  return Buffer.concat(chunks)
}

/** Headers that must never be forwarded verbatim between proxy hops. */
const HOP_BY_HOP_REQUEST_HEADERS = new Set(['host', 'content-length', 'connection'])
const HOP_BY_HOP_RESPONSE_HEADERS = new Set(['content-length', 'content-encoding', 'transfer-encoding', 'connection'])

function forwardableRequestHeaders(rawHeaders, overrideContentLength) {
  const headers = {}
  for (const [key, value] of Object.entries(rawHeaders)) {
    if (value === undefined || HOP_BY_HOP_REQUEST_HEADERS.has(key.toLowerCase())) continue
    headers[key] = value
  }
  if (overrideContentLength !== undefined) headers['content-length'] = String(overrideContentLength)
  return headers
}

function forwardableResponseHeaders(fetchHeaders) {
  const headers = {}
  for (const [key, value] of fetchHeaders) {
    if (HOP_BY_HOP_RESPONSE_HEADERS.has(key.toLowerCase())) continue
    headers[key] = value
  }
  return headers
}

/**
 * Create (but do not start) the proxy's `http.Server`.
 * @param config.upstreamOrigin - scheme+host+port of the real backend, e.g. `http://10.204.100.243:1234` (no path).
 * @param config.maxImagesPerRequest - images kept per forwarded request.
 * @param config.models - model ids to cap; every other model is forwarded completely untouched. Empty/omitted caps every model.
 * @param config.log - optional `(message: string) => void` for routine per-request diagnostics; silent unless supplied.
 * @param config.onError - optional `(message: string) => void` for proxy failures; defaults to `console.error` so real errors are never silent by default.
 * @returns an unstarted `http.Server`; call `.listen()` yourself.
 */
export function createProxyServer(config) {
  const upstreamOrigin = config.upstreamOrigin.replace(/\/+$/, '')
  const modelFilter = config.models && config.models.length > 0 ? new Set(config.models) : undefined
  const onError = config.onError ?? (message => console.error(message))

  return http.createServer((req, res) => {
    void (async () => {
      const targetUrl = new URL(req.url ?? '/', upstreamOrigin)
      try {
        const rawBody = await readBody(req)
        let forwardBody = rawBody.length > 0 ? rawBody : undefined
        const contentType = req.headers['content-type'] ?? ''

        if (req.method === 'POST' && contentType.includes('application/json') && rawBody.length > 0) {
          let payload
          try {
            payload = JSON.parse(rawBody.toString('utf8'))
          } catch {
            payload = undefined
          }
          const inScope = payload !== undefined && (modelFilter === undefined || modelFilter.has(payload.model))
          if (inScope && Array.isArray(payload.messages)) {
            const before = payload.messages
            const after = capImagesInMessages(before, config.maxImagesPerRequest)
            if (after !== before) {
              payload = { ...payload, messages: after }
              forwardBody = Buffer.from(JSON.stringify(payload))
              config.log?.(`vision-3090-fix: capped images in request for model "${payload.model}" to ${targetUrl.pathname}`)
            }
          }
        }

        const upstreamResponse = await fetch(targetUrl, {
          method: req.method,
          headers: forwardableRequestHeaders(req.headers, forwardBody?.length),
          body: forwardBody,
        })

        res.writeHead(upstreamResponse.status, forwardableResponseHeaders(upstreamResponse.headers))
        if (upstreamResponse.body === null) {
          res.end()
          return
        }
        Readable.fromWeb(upstreamResponse.body).pipe(res)
      } catch (error) {
        onError(`vision-3090-fix: proxy error for ${targetUrl}: ${error.message}`)
        if (!res.headersSent) res.writeHead(502, { 'content-type': 'application/json' })
        res.end(JSON.stringify({
          error: { message: `vision-3090-fix proxy error: ${error.message}`, code: 'PROXY_ERROR' },
        }))
      }
    })()
  })
}
