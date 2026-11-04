import { useEffect, useRef, useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { ArrowUpRight, Clapperboard, Disc3, Send, Square } from 'lucide-react'
import { Streamdown } from 'streamdown'

import { useAIChat } from '@/lib/ai-hook'
import type { ChatMessages } from '@/lib/ai-hook'
import {
  recommendInputSchema,
  recommendOutputSchema,
  withLink,
  type Recommendation,
} from '@/lib/recommendations'

type Mode = 'both' | 'movies' | 'songs'

const MODES: { id: Mode; label: string }[] = [
  { id: 'both', label: 'Movies + Songs' },
  { id: 'movies', label: 'Movies' },
  { id: 'songs', label: 'Songs' },
]

const FOCUS_MARKER = /\n\n\[Focus: (movies|songs) only\]$/

const SUGGESTIONS = [
  'A rainy Sunday — something cozy and a little melancholy',
  'I loved Interstellar. What should I watch and listen to next?',
  'Songs that sound like a Wong Kar-wai film',
  'Upbeat 80s tracks for a road trip',
  'Mind-bending thrillers like Memento',
  'Movies that match the vibe of Radiohead’s OK Computer',
]

function parseRecommendations(
  part: { input?: unknown; output?: unknown; arguments: string },
): { heading: string; items: Recommendation[] } | null {
  let output = part.output
  if (typeof output === 'string') {
    try {
      output = JSON.parse(output)
    } catch {
      output = undefined
    }
  }
  const fromOutput = recommendOutputSchema.safeParse(output)
  if (fromOutput.success) return fromOutput.data
  let input = part.input
  if (input === undefined) {
    try {
      input = JSON.parse(part.arguments)
    } catch {
      return null
    }
  }
  const fromInput = recommendInputSchema.safeParse(input)
  if (!fromInput.success) return null
  return {
    heading: fromInput.data.heading,
    items: fromInput.data.items.map(withLink),
  }
}

function RecommendationCard({
  item,
  index,
}: {
  item: Recommendation
  index: number
}) {
  const isMovie = item.kind === 'movie'
  const accent = isMovie ? 'text-reel' : 'text-rhythm'
  const Icon = isMovie ? Clapperboard : Disc3
  return (
    <a
      href={item.link}
      target="_blank"
      rel="noopener noreferrer"
      style={{ animationDelay: `${index * 60}ms` }}
      className={`rise group relative flex flex-col gap-2 rounded-xl border border-line bg-panel p-4 transition hover:-translate-y-0.5 ${
        isMovie ? 'hover:border-reel/60' : 'hover:border-rhythm/60'
      }`}
    >
      <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.18em]">
        <span className={`flex items-center gap-1.5 font-semibold ${accent}`}>
          <Icon className="h-3.5 w-3.5" />
          {isMovie ? 'Film' : 'Track'}
          {item.genre ? (
            <span className="text-muted font-normal">· {item.genre}</span>
          ) : null}
        </span>
        {item.year ? <span className="text-muted">{item.year}</span> : null}
      </div>
      <div>
        <h4 className="font-display text-xl leading-tight text-paper">
          {item.title}
        </h4>
        {item.creator ? (
          <p className="text-sm text-muted">
            {isMovie ? 'dir. ' : ''}
            {item.creator}
          </p>
        ) : null}
      </div>
      <p className="text-sm leading-relaxed text-paper/80">{item.reason}</p>
      <span
        className={`mt-auto inline-flex items-center gap-1 pt-1 text-xs font-semibold ${accent}`}
      >
        Open on {item.linkLabel}
        <ArrowUpRight className="h-3.5 w-3.5 transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
      </span>
    </a>
  )
}

function RecommendationSet({
  heading,
  items,
}: {
  heading: string
  items: Recommendation[]
}) {
  const movies = items.filter((i) => i.kind === 'movie')
  const songs = items.filter((i) => i.kind === 'song')
  return (
    <section className="my-3">
      <h3 className="mb-3 font-display text-lg italic text-paper">
        {heading}
      </h3>
      {[movies, songs].map((group, g) =>
        group.length ? (
          <div
            key={g}
            className="mb-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3"
          >
            {group.map((item, i) => (
              <RecommendationCard key={`${item.title}-${i}`} item={item} index={i} />
            ))}
          </div>
        ) : null,
      )}
    </section>
  )
}

function Messages({ messages }: { messages: ChatMessages }) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [messages])

  return (
    <div ref={containerRef} className="min-h-0 flex-1 overflow-y-auto pb-6">
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-4 pt-6">
        {messages.map((message) => {
          if (message.role === 'user') {
            return (
              <div key={message.id} className="rise flex justify-end">
                <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-paper px-4 py-2.5 text-ink">
                  {message.parts.map((part, index) => {
                    if (part.type !== 'text') return null
                    const focus = part.content.match(FOCUS_MARKER)?.[1]
                    return (
                      <div key={index}>
                        <p className="whitespace-pre-wrap text-[15px]">
                          {part.content.replace(FOCUS_MARKER, '')}
                        </p>
                        {focus ? (
                          <span className="mt-1 inline-block text-[11px] uppercase tracking-widest text-ink/60">
                            {focus} only
                          </span>
                        ) : null}
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          }
          return (
            <div key={message.id} className="flex items-start gap-3">
              <div className="mt-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full border border-line bg-panel">
                <Disc3 className="spin-slow h-4 w-4 text-reel" />
              </div>
              <div className="min-w-0 flex-1">
                {message.parts.map((part, index) => {
                  if (part.type === 'text' && part.content) {
                    return (
                      <div
                        key={index}
                        className="prose prose-invert prose-sm max-w-none text-paper/90"
                      >
                        <Streamdown>{part.content}</Streamdown>
                      </div>
                    )
                  }
                  if (part.type === 'tool-call' && part.name === 'recommend') {
                    const set = parseRecommendations(part)
                    if (!set)
                      return (
                        <p
                          key={index}
                          className="my-3 animate-pulse text-sm text-muted"
                        >
                          Pulling picks from the shelves…
                        </p>
                      )
                    return <RecommendationSet key={index} {...set} />
                  }
                  return null
                })}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Home() {
  const [input, setInput] = useState('')
  const [mode, setMode] = useState<Mode>('both')
  const { messages, sendMessage, isLoading, stop, error } = useAIChat()

  const send = (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || isLoading) return
    void sendMessage(
      mode === 'both' ? trimmed : `${trimmed}\n\n[Focus: ${mode} only]`,
    )
    setInput('')
  }

  return (
    <div className="flex h-dvh flex-col">
      <header className="border-b border-line/70">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-4">
          <div className="flex items-baseline gap-2">
            <span className="font-display text-2xl font-semibold tracking-tight">
              Reel <span className="italic text-reel">&amp;</span> Rhythm
            </span>
            <span className="hidden text-xs uppercase tracking-[0.2em] text-muted sm:inline">
              films · music · moods
            </span>
          </div>
          <div
            role="radiogroup"
            aria-label="Recommendation focus"
            className="flex rounded-full border border-line bg-panel p-1 text-xs"
          >
            {MODES.map((m) => (
              <button
                key={m.id}
                role="radio"
                aria-checked={mode === m.id}
                onClick={() => setMode(m.id)}
                className={`rounded-full px-3 py-1.5 font-medium transition ${
                  mode === m.id
                    ? m.id === 'songs'
                      ? 'bg-rhythm text-ink'
                      : 'bg-reel text-ink'
                    : 'text-muted hover:text-paper'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {messages.length === 0 ? (
        <main className="flex flex-1 items-center justify-center overflow-y-auto px-4 py-10">
          <div className="rise mx-auto w-full max-w-3xl">
            <p className="mb-4 flex items-center gap-2 text-xs uppercase tracking-[0.25em] text-muted">
              <Clapperboard className="h-4 w-4 text-reel" /> IMDb
              <span className="text-line">/</span>
              <Disc3 className="h-4 w-4 text-rhythm" /> Spotify
            </p>
            <h1 className="font-display text-5xl leading-[1.02] font-semibold tracking-tight sm:text-7xl">
              Tell me a mood.
              <br />
              <span className="italic text-reel">I’ll</span> cue the{' '}
              <span className="italic text-rhythm">soundtrack</span>.
            </h1>
            <p className="mt-5 max-w-xl text-lg text-paper/70">
              Describe a feeling, a favorite film, or an artist you can’t stop
              playing. Get movie picks linked to IMDb and songs linked to
              Spotify — or both, paired by vibe.
            </p>
            <div className="mt-8 flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-full border border-line bg-panel/70 px-4 py-2 text-left text-sm text-paper/80 transition hover:border-reel/60 hover:text-paper"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        </main>
      ) : (
        <Messages messages={messages} />
      )}

      <div className="border-t border-line/70 bg-ink/80 backdrop-blur">
        <div className="mx-auto w-full max-w-4xl px-4 py-4">
          {error && (
            <p role="alert" className="mb-3 text-sm text-red-300">
              {error.message}
            </p>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault()
              send(input)
            }}
            className="flex items-center gap-2 rounded-2xl border border-line bg-panel p-2 focus-within:border-reel/60"
          >
            <input
              type="text"
              aria-label="What are you in the mood for?"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                mode === 'movies'
                  ? 'A movie for tonight…'
                  : mode === 'songs'
                    ? 'Songs that feel like…'
                    : 'What are you in the mood for?'
              }
              className="w-full bg-transparent px-3 py-2 text-[15px] text-paper placeholder:text-muted focus:outline-none"
              disabled={isLoading}
            />
            {isLoading ? (
              <button
                type="button"
                onClick={stop}
                aria-label="Stop"
                className="rounded-xl border border-line p-3 text-paper hover:bg-line"
              >
                <Square className="h-4 w-4 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                aria-label="Send message"
                disabled={!input.trim()}
                className="rounded-xl bg-reel p-3 text-ink transition hover:brightness-110 disabled:opacity-40"
              >
                <Send className="h-4 w-4" />
              </button>
            )}
          </form>
          <p className="mt-2 text-center text-[11px] text-muted">
            Picks come from an AI model’s knowledge; links open IMDb and
            Spotify search.
          </p>
        </div>
      </div>
    </div>
  )
}

export const Route = createFileRoute('/')({
  component: Home,
})
