# dsh-vision-3090-fix

A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`) plugin that fixes:

```
400: {"message":"At most 1 image(s) may be provided in one prompt. (parameter=image)","type":"BadRequestError","param":"image","code":400}
```

against a self-hosted OpenAI-compatible vision backend (e.g. [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090), a single-RTX-3090 vLLM deployment). vLLM serves these models with `--limit-mm-per-prompt image=1`, so **any** request carrying more than one image content part is rejected with a 400 — even across unrelated turns of the same conversation, and even across separate tool calls (e.g. a screenshot tool and a file-pull tool each returning one image) in the same turn.

## Why this happens

Harness's shipped LLM adapters (`dsh-llm-pi-ai`, `dsh-llm-deepseek`) only offload images by **accumulated byte size** (`maxRequestImageBytes` / `requestImageMaxBytes`). They never cap by **count**. So a conversation where you attach one small image, get a reply, and then attach a second small image sends **both** images in the next request — comfortably under any byte budget, but two images, which this backend refuses outright.

## What this plugin does

It's a small local HTTP reverse proxy, started as an ordinary Cordis plugin effect (see `docs/user/develop/basic/index.md#automatic-cleanup` in the harness docs — `ctx.effect()` starts and stops it with the plugin's lifecycle). You put it **in front of** your existing provider's `baseURL`; everything else about how you already talk to the backend — `dsh-llm-pi-ai`'s `pure` provider, its model list, its credential — stays exactly as configured.

For each forwarded request, the proxy:

1. Parses the JSON body's `messages` array (standard OpenAI wire format).
2. Counts every `image_url` content part across the whole array, including ones nested in `tool`-role messages (a returned screenshot, a pulled file).
3. Replaces every one beyond the newest `maxImagesPerRequest` (default `1`) with a stable text placeholder, in place.
4. Forwards the request — headers (including `Authorization`, untouched — the proxy never needs or sees your API key's meaning, just passes it through) and the rewritten body — to the real backend.
5. Streams the response straight back, byte for byte, so SSE streaming works exactly as it would talking to the backend directly (verified: chunks arrive incrementally, not buffered).

A request already at or under the cap is forwarded completely unmodified.

## Install

```sh
dsh plugin --profile web add /path/to/dsh-vision-3090-fix
```

(or `dsh plugin --profile web add github:pureexe/dsh-vision-3090-fix` once pushed).

Configure it in your profile's `cordis.patch.yml` (e.g. `~/.dsh/profiles/web/cordis.patch.yml`):

```yaml
- id: vision-3090-fix
  name: dsh-vision-3090-fix
  config:
    upstreamOrigin: http://10.204.100.243:1234   # scheme+host+port only, no path
    listenHost: 127.0.0.1
    listenPort: 8931
    maxImagesPerRequest: 1                        # match your server's --limit-mm-per-prompt
    models: [qwen3.8-27b]                         # optional; omit to cap every model
```

`baseURL` in `dsh-llm-pi-ai` is set once **per provider**, not per model — every model listed under that one provider shares it. So if `pure` serves three models and you point its `baseURL` at this proxy, all three now go through the proxy, even though only one of them needs the cap. `models` (optional; empty means "cap everything") scopes *the cap itself*, not the routing: out-of-scope requests still take the extra local hop through the proxy, but are forwarded completely untouched — same bytes in, same bytes out, no behavior change from talking to the backend directly.

Then point your **existing** provider config at the proxy instead of the real backend — the only line that changes. For a `dsh-llm-pi-ai` route in `settings.yaml`:

```yaml
llm-pi-ai:
  providers:
    pure:
      displayName: pure
      apiKeyEnv: PURE_API_KEY
      api: openai-completions
      baseURL: http://127.0.0.1:8931/v1   # was: http://10.204.100.243:1234/v1
      models:
        - id: qwen3.8-27b
          # ...unchanged
```

Everything else — credentials, model list, `agent-default-model`, the Web UI's Models settings page — keeps working exactly as it did before, because `dsh-llm-pi-ai` still owns the `pure` route and is still the thing editing/reading that section. The proxy is invisible to it beyond the URL.

## Configuration reference

| Field | Default | Meaning |
|---|---|---|
| `upstreamOrigin` | *(required)* | Scheme+host+port of the real backend, e.g. `http://10.204.100.243:1234` — no path |
| `listenHost` | `127.0.0.1` | Host the proxy listens on |
| `listenPort` | *(required)* | Port the proxy listens on; point your provider's `baseURL` at `http://<listenHost>:<listenPort>/v1` |
| `maxImagesPerRequest` | `1` | Images kept per forwarded request; excess (oldest first) becomes placeholder text |
| `models` | `[]` (every model) | Model ids the cap applies to (matched against the request's `model` field); every other model is forwarded byte-for-byte untouched |
| `verbose` | `false` | Log the startup banner and each request that gets capped. Actual proxy errors (e.g. the upstream is unreachable) are always logged regardless — they aren't routine noise. |

## Testing

```sh
npm install
npm test                # unit + local end-to-end tests (fake upstream, no network)
```

To also run the end-to-end test against the real backend:

```sh
VISION_3090_FIX_LIVE_UPSTREAM=http://10.204.100.243:1234 \
VISION_3090_FIX_LIVE_API_KEY=<your-api-key> \
VISION_3090_FIX_LIVE_MODEL=qwen3.8-27b \
VISION_3090_FIX_LIVE_IMAGE=/home/pakkapon/a.png \
node --test test/live.test.js
```

That test starts a real instance of the proxy, first proves an uncapped two-image conversation gets the reported 400 from the real backend, then proves the same conversation succeeds once proxied through a 1-image cap.

## Known limitations

- Buffers the request body fully before forwarding (needed to parse and rewrite JSON); fine for chat/vision payloads, not meant for large file uploads. The response is streamed through without buffering.
- No retry logic and no request queuing — it's a thin pass-through, not a load balancer.
- Assumes the backend is plain HTTP/HTTPS `chat/completions`-shaped JSON; a provider using a different wire shape for images (not OpenAI's `image_url` content part) won't be recognized.

## License

MIT
