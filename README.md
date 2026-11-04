# Reel & Rhythm

A conversational recommendation chatbot for **movies (IMDb)** and **songs (Spotify)**. Describe a mood, an occasion, a favorite film, or an artist, and the assistant replies with recommendation cards: films link to IMDb, tracks link to Spotify. A focus toggle restricts picks to movies only, songs only, or both paired by vibe.

## Tech stack

- [TanStack Start](https://tanstack.com/start) (React 19, file-based routing, server routes)
- [TanStack AI](https://tanstack.com/ai) for streaming chat and server-side tool calls
- Netlify AI Gateway (Anthropic `claude-haiku-4-5` by default; OpenAI and Gemini also supported). No API keys needed on Netlify.
- Tailwind CSS v4, lucide-react icons, Streamdown for markdown

## How it works

The model calls a server tool, `recommend`, with structured items (kind, title, director/artist, year, genre, reason). The server adds an IMDb or Spotify search link to each item, and the UI renders the result as cards. Recommendations come from the model's own knowledge; the app does not call the IMDb or Spotify APIs, so it shows no live ratings or stream counts.

## Run locally

Requires Node 22.13+ (or 24/26) and a linked Netlify project for AI Gateway credentials.

```bash
pnpm install
netlify dev            # or: pnpm dev
pnpm test              # mocked provider tests
```
