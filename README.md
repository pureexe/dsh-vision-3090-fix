# dsh-vision-3090-fix

A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`) plugin that fixes:

```
400: {"message":"At most 1 image(s) may be provided in one prompt. (parameter=image)","type":"BadRequestError","param":"image","code":400}
```

against a self-hosted OpenAI-compatible vision backend (e.g. [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090), a single-RTX-3090 vLLM deployment). vLLM serves these models with `--limit-mm-per-prompt image=1`, so **any** request carrying more than one image content part is rejected with a 400 — even across unrelated turns of the same conversation.

## Why this happens

Harness's shipped LLM adapters (`dsh-llm-pi-ai`, `dsh-llm-deepseek`) only offload images by **accumulated byte size** (`maxRequestImageBytes` / `requestImageMaxBytes`). They never cap by **count**. So a two-turn conversation where you attach one small image, get a reply, and then attach a second small image sends **both** images in the third request — comfortably under any byte budget, but two images, which this backend refuses outright.

This was confirmed directly against the backend (see [`test/live.test.js`](test/live.test.js)):

```
$ curl -X POST http://10.204.100.243:1234/v1/chat/completions ... # 1 image  -> 200 OK
$ curl -X POST http://10.204.100.243:1234/v1/chat/completions ... # 2 images -> 400 "At most 1 image(s) may be provided in one prompt."
```

## What this plugin does

It's a small, self-contained `LlmAdapter` (see [`docs/user/develop/practice/llm-adapter.md`](https://deepseek-harness.github.io/deepseek-harness/) in the harness docs) that speaks the OpenAI-compatible `chat/completions` wire protocol directly:

1. Before building the wire request, it calls Harness's own exported
   `offloadRequestImagesWithPolicy()` (from `@deepseek-ai/dsh-llm`) with a
   **count** budget — `maxImages: maxImagesPerRequest` (default `1`) — keeping
   only the newest image(s) and replacing every older one with the same
   stable placeholder text (`offloadedImageText`) the built-in byte-based
   offload uses. This is the one behavior gap it closes; everything else is a
   direct, minimal implementation of the wire protocol so the fix has as
   little surface area as possible.
2. It converts the resulting history into OpenAI-compatible `messages`
   (text, `image_url` data URIs, assistant `tool_calls`, `tool` role
   messages).
3. It streams the response (SSE) and translates it into Harness's
   `StreamChunk` protocol, including `reasoning_content`, tool calls, and
   usage (mirroring the same wire shapes `dsh-llm-deepseek` handles).

## Install

This plugin **replaces** the adapter for one provider route — it cannot be mounted alongside another adapter that registers the same route name (Harness rejects duplicate route registration). If you currently reach this backend through `dsh-llm-pi-ai`'s `pure` provider (as in the original bug report), remove that provider block from your `llm-pi-ai:` settings section first.

### 1. Add the plugin to your profile

```sh
dsh plugin --profile web add /path/to/dsh-vision-3090-fix
```

(or `dsh plugin --profile web add github:<you>/dsh-vision-3090-fix` once pushed).

### 2. Configure it

Edit your profile's `cordis.patch.yml` (e.g. `~/.dsh/profiles/web/cordis.patch.yml`) — this bundle's own `cordis.patch.yml` ships a placeholder row; override it by `id`, restating the whole config:

```yaml
- id: vision-3090-fix
  name: dsh-vision-3090-fix
  config:
    providers: [pure]                       # route name(s) this adapter owns
    baseURL: http://10.204.100.243:1234/v1
    apiKeyEnv: PURE_API_KEY                  # set this env var before starting dsh
    maxImagesPerRequest: 1                   # match your server's --limit-mm-per-prompt
    models:
      - id: qwen3.8-27b
        name: qwen3.8-27b
        contextWindow: 131072
        maxTokens: 16384
        input: [text, image]
        reasoningEfforts:
          low: low
          medium: medium
          high: high
          xhigh: xhigh
        defaultReasoningEffort: medium
```

Set the referenced environment variable (`PURE_API_KEY` above) before launching `dsh`. A literal `apiKey: "..."` field also works, but keeping the secret out of `cordis.yml`/patch files is preferred.

### 3. Point your default model at the route (if needed)

`agent-default-model.provider: pure` in `settings.yaml` keeps working unchanged, since routing is by route name, not by which adapter owns it.

## Configuration reference

| Field | Default | Meaning |
|---|---|---|
| `providers` | `['pure']` | Route name(s) this adapter registers for |
| `baseURL` | *(required)* | Base URL of the OpenAI-compatible server |
| `apiKey` | — | Literal API key |
| `apiKeyEnv` | — | Env var read once at plugin load for the API key |
| `maxImagesPerRequest` | `1` | Images kept per request; excess (oldest first) becomes placeholder text |
| `defaultContextWindow` | `131072` | Context window for a model id absent from `models` |
| `defaultMaxTokens` | `16384` | Output cap for a model id absent from `models` |
| `requestImagePixelBudget` | `4194304` | Total-pixel budget applied when resolving one image's request bytes |
| `requestImageMaxBytes` | `1048576` | Encoded-byte target applied when resolving one image's request bytes |
| `models` | `[]` | Static model catalog: `id`, `name`, `contextWindow`, `maxTokens`, `input`, `reasoningEfforts`, `defaultReasoningEffort` |

## Testing

```sh
npm install
npm test                # unit tests only (no network)
```

To also run the end-to-end test against a real backend:

```sh
VISION_3090_FIX_LIVE_BASE_URL=http://10.204.100.243:1234/v1 \
VISION_3090_FIX_LIVE_API_KEY=<your-api-key> \
VISION_3090_FIX_LIVE_MODEL=qwen3.8-27b \
VISION_3090_FIX_LIVE_IMAGE=/home/pakkapon/a.png \
node --test test/live.test.js
```

That test first reproduces the reported 400 with an uncapped request, then proves the same two-image conversation succeeds once capped to 1 image.

## Known limitations

- Images returned by a tool call (a screenshot, a pulled file) are treated the same as a directly attached image and count toward `maxImagesPerRequest`; they are sent as real `image_url` content inside the `tool` role message, which this specific backend accepts, but this is not a universal part of the OpenAI `tool` message spec — verify it against your own server if you rely on it.
- `reasoning_effort` is sent as a best-effort `reasoning_effort` wire field using each model's configured spelling; there is no universal OpenAI-compatible convention for this, so verify it against your server's chat template.
- No retry logic — a transient provider failure surfaces once as an `LlmError`, same contract every Harness adapter is expected to meet (`dsh-llm-retry` re-runs failed requests at the agent-step boundary if mounted).
- `GenerateOptions.stop` is passed straight through as `stop`; not every backend honors it identically.

## License

MIT
