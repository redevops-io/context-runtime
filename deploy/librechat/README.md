# LibreChat × Context Runtime — self-learning RAG chat

LibreChat, wired to the Context Runtime control plane so that **every chat turn is a
self-learning retrieval loop**: Context Runtime plans and retrieves context for the
user's message (using a *learned* retrieval strategy), injects it, answers (via an
upstream model or from the context), then **judges the retrieval quality and learns**
the best strategy for that kind of request. The benchmark is retrieval quality vs. the
user request — exactly the fleet pattern, with LibreChat as the tenant.

## How it fits together

```
LibreChat (custom OpenAI endpoint)  ──►  Control plane /v1/chat/completions (the shim)
                                             │  1. /librechat/retrieve  (learned strategy)
                                             │  2. inject context, forward to a model
                                             │  3. judge retrieval quality → learn
                                             ▼
                                    Context Runtime fleet (bandit + cost model, persisted)
```

The shim is `context_runtime/control_plane/app.py` → `GET /v1/models`,
`POST /v1/chat/completions` (streaming + non-streaming). The OpenAI response carries a
`context_runtime` block (`strategy`, `retrieval_score`, `reward`, `citations`,
`suggestion`).

## 1. Build a corpus (multimodal)

```bash
python -m context_runtime.ingest.multimodal /path/to/files /path/to/messages --out ./corpus
# text-layer (PDF/DOCX/HTML) needs `pip install -e '.[ingest]'`;
# image OCR + audio/video ASR need `.[ingest-multimodal]` (rapidocr + faster-whisper).
```

## 2. Run the control plane (with the corpus + optional upstream model)

```bash
pip install -e '.[control-plane]'
CR_CORPUS_DIR=./corpus \
CONTEXT_RUNTIME_API_KEY=context-runtime \
# optional: forward to a real OpenAI-compatible model instead of answering from context
CR_UPSTREAM_BASE_URL=https://api.moonshot.ai/v1 CR_UPSTREAM_API_KEY=$KIMI_API_KEY CR_UPSTREAM_MODEL=kimi-k2.6 \
uvicorn context_runtime.control_plane.app:app --host 0.0.0.0 --port 8091
```

## 3. Run LibreChat, pointed at the control plane

```bash
cd deploy/librechat
CONTEXT_RUNTIME_URL=http://host.docker.internal:8091 \
CONTEXT_RUNTIME_API_KEY=context-runtime \
docker compose up -d
```

Open http://localhost:3080, register, pick **"Context Runtime (self-learning RAG)"**,
and chat. Watch the policy learn: `GET http://localhost:8091/librechat/policy`.

The same custom endpoint is also registered in the full fork config
(`LibreChat/librechat.yaml`) for deploying alongside its other endpoints.

## Retrieval-comparison panel (transparency)

The fork adds a **Context Runtime panel under the chat box** that, for each request,
shows what BM25 / vector / hybrid / community / graph retrieval each return side by side
and highlights the strategy the learned policy actually served — making the "the runtime
picks the best method" thesis visible to users.

- Frontend: `LibreChat/client/src/components/Chat/Input/RetrievalCompare.tsx`, rendered by
  `ChatView.tsx` under `<ChatForm>`.
- Data: read-only `POST /librechat/compare {request,k}` on the control plane (both the
  Python and Go runtimes implement it). CORS is already open on both.
- Because it's a frontend change, LibreChat must be **built from the fork** — the compose
  now uses `build:` instead of the prebuilt image (`docker compose up -d --build`).
- The panel calls `http://localhost:8092/librechat/compare` by default (the browser runs
  on the host). To point it at the Go runtime (8093) or a remote host, set
  `VITE_CR_COMPARE_URL` before building the frontend (or edit the default in
  `RetrievalCompare.tsx`).

## Cross-language search (CR_QUERY_LANGS)

By default retrieval is same-language — an English question won't match a Russian corpus
(no retrieval method bridges languages). Opt in on the control plane:

```bash
CR_QUERY_LANGS=ru                     # corpus language(s); "ru" or "ru,en"
CR_QUERY_XLATE_MODEL=moonshot-v1-8k   # a FAST model for translation (reasoning models take 60s+)
```

The runtime translates the query's key terms into those languages before retrieving, so
"any cholesterol results?" surfaces the Russian lipid PDFs (холестерин/липидный профиль).
Translations are cached (first query per phrase pays the LLM latency, then instant) and
fail-open. Works for both the chat and the Libre Query Board, on both runtimes.
