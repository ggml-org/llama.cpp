# @ggml-org/llama-ui

Llama UI, the design system of Llama Chat and the Llama apps, as a package: its tokens, fonts, surfaces and
components, for Llama Chat (`ui/apps/chat`) and llama.pages (llama.app). It lives in llama.cpp at `ui/lib/ui`,
beside the apps built on it in `ui/apps`.

## Use

```css
/* app.css */
@import 'tailwindcss';
@import 'tw-animate-css';
@import '@ggml-org/llama-brand/css/colors.css';
@import '@ggml-org/llama-ui/theme.css';
```

```svelte
<script lang="ts">
	import { Badge, Button } from '@ggml-org/llama-ui';
</script>

<Button variant="secondary">View on Hugging Face</Button>
<Badge variant="success">Downloaded</Badge>
```

- `theme.css`: everything: fonts, tokens, surfaces, the Tailwind theme and base styles. It needs Tailwind 4,
  `tw-animate-css` (both peer dependencies) and `@ggml-org/llama-brand`'s `css/colors.css` imported before it.
- `tokens.css`: the three tiers of tokens only, also after `@ggml-org/llama-brand`'s `css/colors.css`. `fonts.css`: Llama Sans and Llama Mono. `surfaces.css`: the matte
  surface utilities.
- Dark mode is the `dark` class on `<html>`.

## What's in it

- **Foundations:** the primitive, semantic and component tokens, the type roles (`text-body`, `text-label`…), corner,
  icon and control-height roles, the matte surfaces and elevations, Llama Sans and Llama Mono.
- **Components:** every base component of Llama Chat: Alert, AlertDialog, Badge, Button, ButtonGroup, Card, Checkbox,
  Collapsible, ContextMenu, Dialog, Drawer, DropdownMenu, Empty, HoverCard, Input, Label, Popover, RadioGroup,
  ScrollArea, Select, Separator, Skeleton, Switch, Table, Tabs, Textarea, Toggle, ToggleGroup and Tooltip. Each is
  also its own entry point (`@ggml-org/llama-ui/ui/dialog`), with `cn` and `tv` at `@ggml-org/llama-ui/utils`. Then ActionIcon (an icon button with a tooltip), SearchInput (a search field with a clear button), NativeSelect (the library's select trigger over the OS's own picker, for touch screens), SyntaxHighlightedCode (code as Llama Chat shows it, in highlight.js's GitHub themes; needs `mode-watcher`) and Logo (the brand mark, llama-brand's icon), at `@ggml-org/llama-ui/app/actions`, `app/forms`, `app/content` and `app/misc`. A phone's sheets are built from `app/sheets`: `SheetHeader` (the close or back key and the title) and `SheetPages` with `useSheetPages` (a stack of pages that slide) inside a Drawer whose content takes `SHEET_CONTENT_CLASS`, and `InsetList`, `InsetListRow` and `InsetListSplitRow` for their native-style lists. `FILTER_TRIGGER_CLASS` and `FILTER_TOGGLE_ITEM_CLASS` style a filters bar the way Llama Chat's Models Manager does.
- **App mockups:** MacMenu (Llama for Mac's menu) and WindowsTray (Llama for Windows' tray flyout), the native apps' UIs as the brand's communications draw them, with an `annotate` snippet to pin a page's numbered callouts to their edges.
- **Model UI:** how a model is shown in Llama Chat and on llama.app: `ModelCard`, `ModelFamilyCard` (a family as llama.app's catalog
  shows it), `ModelId` (name, badges, capability,
  context and size icons), `ModelCapabilityIcons` (icons or labelled badges), `ModelOrgAvatar`, `ModelMetadataItem`,
  `ModelQuantOptions`, `ModelQuantChip`, `ModelQuantRow`, `ModelDraftSidecars`, `ModelDownloadProgressBar`, plus the formatters and
  `minMemoryTierGb` and `summarizeSizes`, at `@ggml-org/llama-ui/models`. They take model data in and report what the user did out;
  fetching and downloading stay in the app.
- **Voice UI** (also `@ggml-org/llama-ui/voice`): `VoiceWaveformLive` (the waveform while recording), `VoiceWaveform` (a
  recorded clip: play, seek, time) and `VoiceTranscriptReveal` (a clip with its transcript, opened on hover, keyboard
  focus or a tap), plus `formatClipTime` and the waveform math. They take levels, peaks, an audio URL and a transcript
  in; recording and transcribing stay in the app.

Next: llama.app's model page.

The constants, enums and code helpers the components use are entry points too, for an app that shows the same things:
`@ggml-org/llama-ui/constants` (the shared classes and code-block constants), `@ggml-org/llama-ui/enums` (`ColorMode`,
`TooltipSide`, `BooleanString`) and `@ggml-org/llama-ui/code` (`highlightCode`, `trimCodePadding`,
`loadHighlightLanguages`, `isHighlightLanguageLoaded`, `resolveHighlightLanguage`, `getLoadedHighlightLanguages`).
highlight.js loads one language at a time, each its own chunk, so none is in an app's first load: until a language
has loaded, `highlightCode` returns the code escaped as plain text, and a `$derived` or `$effect` that called it runs
again once it has. A language loads with the ones it embeds (CSS and JavaScript inside HTML); auto-detection picks
among lowlight's common languages. Await `loadHighlightLanguages([...])` where highlighted output is needed at once.
`scripts/generate-highlight-languages.mjs` regenerates the language table after a highlight.js or lowlight update. The model data
(constants, enums, formatters and types) comes with the Model UI at `@ggml-org/llama-ui/models`.

## How the apps in this repo use it

Llama Chat does not install the package: it compiles these sources as part of its own build, so a change here
shows in the app at once and the app keeps no copy of a component or a token. For that to work:

- Sources under `src/lib` import each other by relative path, never through `$lib`, which in an app points at
  the app's own `src/lib`.
- What they import from npm is listed in `dependencies` or `peerDependencies`: the app reads that list to resolve
  those packages from its own `node_modules` (`ui/apps/chat/scripts/llama-ui.js`).
- Stylesheets import no package. An app imports Tailwind, `tw-animate-css` and `@ggml-org/llama-brand`'s `css/colors.css`
  (the brand's colours, under the names the primitives use) itself, before `theme.css`.
- A component added with shadcn-svelte (`components.json`) arrives with `$lib` imports: make them relative.

## Development

- `npm run storybook`: the package's Storybook, with a story for everything it holds; the stories run as tests in
  `npm test`, and Llama Chat's Storybook (in `ui/apps/chat`) shows them under Llama UI
- `npm run dev`: Storybook and a small SvelteKit check page (`src/routes`)
- `npm run check`: svelte-check
- `npm run prepack`: build the library into `dist/` and run publint
- `node scripts/npm-version-check.mjs`: what the release workflow would make of this folder, against the real registry

## Releases

The package is published from here by
[`ui-lib-publish.yml`](../../../.github/workflows/ui-lib-publish.yml), which runs when a commit reaches `master` with
`ui/lib/ui` in it. The version in `package.json` is the release: land a bump together with the change that should ship
and the workflow type-checks, builds, lints and publishes that version to npm, with a provenance attestation. A library
change that carries no bump publishes nothing and leaves a warning on the run.

- `npm version <patch|minor> --no-git-tag-version` and commit: it moves `package.json` and both version fields of the
  lockfile together.
- Below `1.0`: a breaking change bumps the minor, a feature bumps the patch.
- A version with a prerelease suffix (`0.5.0-beta.1`) publishes under the `next` tag, so `latest` stays a stable build.
- Publishing needs no token in the repo: the job trades its OIDC identity for one, so the only place to look when a
  publish is refused is the package's trusted-publisher settings on npm. A publisher names a workflow that must already
  exist under `.github/workflows`, so it is added after this workflow lands, and it allows staged publishing until
  `npm publish` is permitted for it. Provenance is generated without asking, and only on a GitHub-hosted runner: a
  self-hosted one cannot publish this way.
