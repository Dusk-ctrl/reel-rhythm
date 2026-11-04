import { createFileRoute } from '@tanstack/react-router'

// Forwards the research demo page (/research/) to the separate Python FastAPI
// backend in backend/. Set RECOMMEND_API_URL (e.g. https://my-host.example.com)
// to enable it; API_ACCESS_TOKEN is forwarded as a bearer token when set.
async function proxyRecommend(request: Request): Promise<Response> {
  const base = process.env.RECOMMEND_API_URL?.trim()
  if (!base) {
    return Response.json(
      { detail: 'The Python research backend is not connected yet (RECOMMEND_API_URL is not set).' },
      { status: 503 },
    )
  }
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const token = process.env.API_ACCESS_TOKEN?.trim()
  if (token) headers.Authorization = `Bearer ${token}`
  try {
    const upstream = await fetch(new URL('/api/recommend', base), {
      method: 'POST',
      headers,
      body: await request.text(),
      signal: AbortSignal.timeout(20_000),
    })
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { 'Content-Type': upstream.headers.get('Content-Type') ?? 'application/json' },
    })
  } catch {
    return Response.json({ detail: 'The Python research backend did not respond.' }, { status: 502 })
  }
}

export const Route = createFileRoute('/api/recommend')({
  server: { handlers: { POST: ({ request }) => proxyRecommend(request) } },
})
