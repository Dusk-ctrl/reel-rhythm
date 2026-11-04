import { z } from 'zod'

export const recommendationItemSchema = z.object({
  kind: z.enum(['movie', 'song']).describe('movie (IMDb) or song (Spotify)'),
  title: z.string().trim().min(1).max(200).describe('Movie or song title'),
  creator: z
    .string()
    .trim()
    .max(200)
    .describe('Director for movies, artist for songs'),
  year: z.number().int().min(1880).max(2100).optional(),
  genre: z.string().trim().max(80).optional().describe('Short genre label'),
  reason: z
    .string()
    .trim()
    .min(1)
    .max(400)
    .describe('One sentence on why this fits the request'),
})

export const recommendationSchema = recommendationItemSchema.extend({
  link: z.string(),
  linkLabel: z.string(),
})

export type Recommendation = z.infer<typeof recommendationSchema>

export const recommendOutputSchema = z.object({
  heading: z.string(),
  items: z.array(recommendationSchema),
})

export const recommendInputSchema = z.object({
  heading: z
    .string()
    .trim()
    .min(1)
    .max(120)
    .describe('Short heading for this set, e.g. "Rainy-night thrillers"'),
  items: z.array(recommendationItemSchema).min(1).max(10),
})

export function imdbLink(title: string, year?: number) {
  const q = year ? `${title} ${year}` : title
  return `https://www.imdb.com/find/?q=${encodeURIComponent(q)}&s=tt&ttype=ft`
}

export function spotifyLink(title: string, artist: string) {
  return `https://open.spotify.com/search/${encodeURIComponent(
    `${title} ${artist}`.trim(),
  )}/tracks`
}

export function withLink(item: z.infer<typeof recommendationItemSchema>): Recommendation {
  return item.kind === 'movie'
    ? { ...item, link: imdbLink(item.title, item.year), linkLabel: 'IMDb' }
    : {
        ...item,
        link: spotifyLink(item.title, item.creator),
        linkLabel: 'Spotify',
      }
}
