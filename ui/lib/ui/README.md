# @ggml/llama-ui

Llama UI, the design system of Llama Chat and the Llama apps, as a package: its tokens, fonts, surfaces and
components, for llama.pages (llama.app) first and Llama Chat (`ui/apps/chat`) later. It lives in llama.cpp at
`ui/lib/ui`, beside the apps built on it in `ui/apps`.

## Use

```css
/* app.css */
@import 'tailwindcss';
@import '@ggml/llama-ui/theme.css';
```

```svelte
<script lang="ts">
	import { Badge, Button } from '@ggml/llama-ui';
</script>

<Button variant="secondary">View on Hugging Face</Button>
<Badge variant="success">Downloaded</Badge>
```

- `theme.css`: everything: fonts, tokens, surfaces, the Tailwind theme and base styles. It needs Tailwind 4.
- `tokens.css`: the three tiers of tokens only. `fonts.css`: Llama Sans and Llama Mono. `surfaces.css`: the matte
  surface utilities.
- Dark mode is the `dark` class on `<html>`; the Matte light surfaces are the `theme-matte` class.

## What's in it

- **Foundations:** the primitive, semantic and component tokens, the type roles (`text-body`, `text-label`…), corner,
  icon and control-height roles, the matte surfaces and elevations, Llama Sans and Llama Mono.
- **Components:** Button, Badge, Card, Input, Label, Popover, Select, Separator, Switch, Toggle, ToggleGroup, Tooltip, ActionIcon (an icon button with a tooltip), SearchInput (a search field with a clear button), SyntaxHighlightedCode (code as Llama Chat shows it, in highlight.js's GitHub themes; needs `mode-watcher`) and Logo (the brand mark). `FILTER_TRIGGER_CLASS` and `FILTER_TOGGLE_ITEM_CLASS` style a filters bar the way Llama Chat's Models Manager does.
- **App mockups:** MacMenu (Llama for Mac's menu) and WindowsTray (Llama for Windows' tray flyout), the native apps' UIs as the brand's communications draw them, with an `annotate` snippet to pin a page's numbered callouts to their edges.
- **Model UI:** how a model is shown in Llama Chat and on llama.app: `ModelCard`, `ModelFamilyCard` (a family as llama.app's catalog
  shows it), `ModelId` (name, badges, capability,
  context and size icons), `ModelCapabilityIcons` (icons or labelled badges), `ModelOrgAvatar`, `ModelMetadataItem`,
  `ModelQuantOptions`, `ModelQuantChip`, `ModelQuantRow`, plus the formatters and
  `minMemoryTierGb` and `summarizeSizes`. They take model data in and report what the user did out; fetching and downloading stay in the
  app.

Next: Tabs and Separator, and llama.app's model page from its proposal (Scrapbook › UI › llama.app).

## Development

- `npm run storybook`: a copy of Llama UI's Storybook in llama.cpp, cut down to what the package holds, with the
  Scrapbook for proposals
- `npm run dev`: Storybook and a small SvelteKit check page (`src/routes`)
- `npm run check`: svelte-check
- `npm run prepack`: build the library into `dist/` and run publint
