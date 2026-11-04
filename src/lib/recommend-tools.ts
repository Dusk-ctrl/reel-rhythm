import { toolDefinition } from '@tanstack/ai'
import {
  recommendInputSchema,
  recommendOutputSchema,
  withLink,
} from './recommendations'

export const recommendToolDef = toolDefinition({
  name: 'recommend',
  description:
    'Show a set of movie and/or song recommendations to the user as rich cards with IMDb and Spotify links. Call this whenever you recommend titles.',
  inputSchema: recommendInputSchema,
  outputSchema: recommendOutputSchema,
})

export const recommend = recommendToolDef.server(({ heading, items }) => ({
  heading,
  items: items.map(withLink),
}))
