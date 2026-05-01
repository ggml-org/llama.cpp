---
name: app
description: Opinionated app components building on top of ./ui primitives
---

- Can include business logic and state management
- Can include data fetching and caching logic
- Should use original spelling for HTML-native events and `camelCase` for custom events
- Props and markup attributes should be listed alphabetically
- Use JS Objects and Arrays for CSS classes and styles when they are dynamic
- Whenever there can be repetition in the component's markup, if it's too small to be decoupled as a separate component — use Svelte 5's `{#snippet}` + `{@render}`
