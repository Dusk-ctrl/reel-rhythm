# Reel & Rhythm: agent notes

A movie + song recommendation chatbot built with TanStack Start, TanStack AI, and Netlify AI Gateway.

## Architecture

- `src/routes/index.tsx`: the whole UI: header with focus toggle (Movies + Songs / Movies / Songs), empty-state hero with suggestion chips, chat messages, and recommendation cards rendered from `recommend` tool-call parts.
- `src/routes/api.chat.ts`: POST `/api/chat`, which delegates to `handleChatRequest`.
- `src/lib/chat.server.ts`: request validation (zod), provider and adapter selection (Anthropic → OpenAI → Gemini by available key; Ollama opt-in), system prompt, SSE streaming, and cancellation.
- `src/lib/recommendations.ts`: shared zod schemas and IMDb/Spotify link builders. It has no `@tanstack/ai` imports, so it stays safe to bundle on the client.
- `src/lib/recommend-tools.ts`: the `recommend` tool definition and its server implementation, which attaches links.
- `src/lib/ai-hook.ts`: client chat connection to `/api/chat` with error surfacing.
- `src/lib/chat.server.test.ts`: vitest tests that mock every provider HTTP call.

## Non-obvious decisions

- **No live IMDb or Spotify APIs.** Picks come from the LLM's knowledge. Links are *search* URLs (`imdb.com/find`, `open.spotify.com/search/.../tracks`), so they work without IDs or keys. Do not show invented ratings or play counts.
- **Focus mode** is sent by appending `\n\n[Focus: movies only]` or `[Focus: songs only]` to the user message. The system prompt honors it, and the UI strips it (`FOCUS_MARKER`) and shows a small chip instead.
- Cards render from the tool-call `output` when present (it includes links). Otherwise they fall back to parsing `input`/`arguments` and building links client-side, so cards appear before the tool result arrives.
- The message schema in `chat.server.ts` only allows tool calls named `recommend`. Update it if you add tools.
- Keep TypeScript 6; TS 7 breaks the Netlify adapter build.
- Do not pass API keys yourself. AI Gateway injects provider env vars at runtime.

## Conventions

- Tailwind v4 theme tokens live in `src/styles.css` (`ink`, `panel`, `line`, `paper`, `muted`, `reel` = amber for movies, `rhythm` = green for songs; fonts Fraunces as display and Instrument Sans as body).
- Site metadata (`siteName`, `siteDescription`) lives in `src/routes/__root.tsx`.

## Python backend

- `backend/` holds separate Python research code (Gemma 4 SFT vs. GRPO on IMDb/Spotify questions, plus a rule-based FastAPI demo). Netlify does not build or run it, and the web app does not call it yet. Run it from `backend/` (`python main.py --mode ...`) so `src.*` imports resolve.
- `public/research/index.html` is a static research demo page served at `/research/`. It POSTs to `/api/recommend` (`src/routes/api.recommend.ts`), which proxies to the FastAPI backend at `RECOMMEND_API_URL` (forwarding `API_ACCESS_TOKEN` as a bearer token) and returns 503 JSON when that variable is unset. Keep it out of `public/index.html`, because `/` is the chatbot.
