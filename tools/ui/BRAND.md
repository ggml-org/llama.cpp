# Llama brand and communications

This is the brand book for **Llama** (the app) and **llama.cpp** (the engine): the rules for posts, posters,
release images, video thumbnails, link previews, banners and any other piece the team makes. It is written for
agents first (Claude Design reads it as the README of the Llama design system) and for people too.

**The Storybook is the source of truth.** Every rule here is drawn there with the real fonts, tokens and
components, under **Communications** (Direction, Foundations, Elements, Modules, Key visuals, Applications),
**Brand** (Logo, Typography, Colors) and **Philosophy**. When this file and the Storybook disagree, the Storybook
wins. This file lives at `tools/ui/BRAND.md` in llama.cpp; the Llama design system in Claude is exported from it
and from the Storybook by `npm run export:design-system`. The how of the app's own UI is `DESIGN.md`.

## What we stand for

Running AI on your own computer is freedom: you own the model, your conversations and your traces, you choose
what runs, and no one changes it behind your back. Open models on the hardware people already have make AI
available to everyone, and a server one person runs can be shared by many. The apps are simple by default and
deep when you want them. We say all of this by showing it, never by preaching it.

### Voice

- Plain words and real numbers. Short sentences. Name things the way people know them.
- Calm and hopeful. No pride, no righteousness, no fear and no hype.
- Show what the app does rather than mock the alternative.
- Name model makers in words, never with their logos. Never pick favourites among models or vendors.

| We say                                     | We don't                                                                          |
| ------------------------------------------ | --------------------------------------------------------------------------------- |
| Runs on your computer, even offline.       | The most powerful AI ever built.                                                  |
| 4.6 GB. Fits in 8 GB of memory.            | Blazing fast. Insanely capable.                                                   |
| Your conversations stay with you.          | They are watching everything you type.                                            |
| 100,000+ open models. One computer: yours. | Or would you rather keep paying for burning tokens in the cloud? Choose wisely ;) |

## Names and lockups

- **Llama** is the product: capital L, set in Llama Sans. It may go on any ground.
- **llama.cpp** is the engine: always lowercase, set in Llama Mono Bold, with `.cpp` in cinder. It goes on the
  neutral grounds only (light, dark).
- Both share one icon, the same glyph in both fonts. Use the lockup and icon files from the assets; never redraw,
  stretch, recolour (beyond ink and light) or outline them. In live text, the lockup is typed: U+E002 followed by
  the name, in the matching font (in Llama Mono at 90% of the size so its icon matches Llama's).

## Type

- **Llama Sans** for everything people read: headlines, ledes, body, captions. One variable font, weights 400 to
  700, with optical sizes from 16 to 40 that the browser picks from the font size. Headlines Bold (700), tight
  (letter-spacing -0.02em to -0.035em), line height 0.92 to 0.98.
- **Llama Mono** only for what is code or data: commands, URLs and repo ids, figures in tables and receipts, and
  small uppercase labels (0.12em tracking). Never for headlines or running text.
- Figures are tabular. A number in a piece is real and states its unit.

Sizes are multiples of the format's unit `u` (see Grid):

| Role     | Size  | Notes                                                                |
| -------- | ----- | -------------------------------------------------------------------- |
| Display  | 3.2u  | Bold, line height 0.92, -0.035em. One statement, two to three lines. |
| Headline | 2u    | Bold, line height 0.98, -0.02em.                                     |
| Figure   | 5u    | Bold. One number that carries the piece.                             |
| Lede     | 0.72u | Regular, in the ground's muted ink.                                  |
| Body     | 0.55u | Regular, line height 1.4.                                            |
| Label    | 0.48u | Llama Mono Medium, uppercase, 0.12em tracking, muted.                |
| Mono     | 0.5u  | Llama Mono, commands and data.                                       |

## Colour

Communications are neutral by default. Colour comes from five grounds, the art colours for artwork, and one
accent.

| Ground    | Background                               | Ink        | Lockups          | For                            |
| --------- | ---------------------------------------- | ---------- | ---------------- | ------------------------------ |
| Light     | `#fafafa`                                | `#111111`  | Llama, llama.cpp | The default                    |
| Dark      | `#111111`                                | `#fafafa`  | Llama, llama.cpp | The developer side, the engine |
| Sand      | `sand-50`                                | `sand-900` | Llama            | A warm, quiet piece            |
| Day sky   | a light-blue gradient, deeper at the top | `#111111`  | Llama            | llama.app's hero, the product  |
| Night sky | a deep blue gradient, darkest at the top | `#fafafa`  | Llama            | The product at night           |

- **Cinder** (`art-cinder`, `oklch(0.68 0.18 46)`, #ee6c1e) is the one accent: one thing per piece, such as the
  chosen path, the figure that matters, a tag, the `.cpp` of the logotype. Never a ground.
- **Art colours** (`art-blue`, `art-sky`, `art-navy`, `art-green`, `art-amber`, `art-cinder-light`) are for
  illustration and artwork details, never for grounds of whole pieces and never for text.
- **No red.** Red is the UI's danger colour only.
- On every ground each piece reads its inks from the ground: ink, muted ink (about 62% of ink), a hairline (about
  10 to 14%), a card and an inset fill. Text keeps 4.5:1 against its ground.

## Formats and grid

Every size in a piece is a multiple of one unit, **u = the short side ÷ 24** (on formats wider than 2:1, the long
side ÷ 48). The margin is 2u on every side, plus the platform's safe area (on the Microsoft Store's 16:9 pieces, the
bottom quarter, where the Store may set its own text). Square and tall formats have 6 columns,
wide ones 12, with a gutter of ½u.

| Format                | Size        | For                                                                   | u     |
| --------------------- | ----------- | --------------------------------------------------------------------- | ----- |
| Square                | 1080 × 1080 | Instagram, LinkedIn, X                                                | 45    |
| Portrait 4:5          | 1080 × 1350 | Instagram, LinkedIn                                                   | 45    |
| Story 9:16            | 1080 × 1920 | Stories, Shorts, TikTok; keep clear of the top 220px and bottom 320px | 45    |
| Landscape 16:9        | 1600 × 900  | X, Bluesky, slides, video thumbnails                                  | 37.5  |
| Open Graph            | 1200 × 630  | Link previews on llama.app                                            | 26.25 |
| GitHub social preview | 1280 × 640  | The repo's link preview                                               | 26.7  |
| Poster A-series       | 1240 × 1754 | Print at 150 dpi                                                      | 51.7  |
| Billboard 3:1         | 2400 × 800  | Banners, screens at events                                            | 50    |
| Super hero art 16:9   | 1920 × 1080 | The top of the Microsoft Store listing; artwork with no words         | 45    |
| Store screenshot 16:9 | 1920 × 1080 | Microsoft Store screenshots: a centred headline over the real app     | 45    |
| Poster art 2:3        | 720 × 1080  | The Microsoft Store's main logo: the icon over the name               | 30    |
| App tile icon         | 300 × 300   | The Microsoft Store's tile: the icon alone                            | 12.5  |

## Building blocks

- **Header**: the lockup (or the icon before a Mono kicker), an optional pill with a cinder dot, the URL in Mono at
  the right.
- **Footer**: who it is from on the left, the promise on the right, in muted ink.
- **Chips**: Mono pills. Outline for use cases, solid for a tag, cinder once.
- **Figure**: a big number with a Mono label under it.
- **Command**: a terminal line in Mono with a cinder `$`, on the ground's inset, always on one line.
- **Model card**: maker and purpose, name, a short description, figures in cinder, Mono facts and the repo id.
- **Stat card**: a figure, its label and what it means.
- **Product shot**: the real Llama app (its menu bar and panel), never a mock-up.
- **Motifs**: line drawings in the ground's ink with one cinder detail. Models and maths: decision tree (decision
  models), neural network, tensor, gradient descent, vector space. Charts and fields: curve, quantised blocks, dot
  grid. Machines, each isometric or flat: GPU, laptop, small workstation, large workstation, server.

## Key visuals

Layouts built from the blocks on the grid. A new piece starts from one of them.

| Key visual | Use it for                                            | Shape                                                  |
| ---------- | ----------------------------------------------------- | ------------------------------------------------------ |
| Statement  | One thing said big: a launch line, a manifesto        | Display headline, lede, a motif filling the rest       |
| Product    | The app itself: link previews, launch images          | Headline beside the product shot, a caption chip       |
| Roster     | Several items of one kind: a model drop               | Headline with a motif, chips, a row of cards           |
| Bulletin   | News with several items: what's new, release notes    | Headline and lede, a lead item and a side column       |
| Numeral    | One figure that carries the point: speed, size, count | The figure huge in cinder over a dot grid              |
| Receipt    | A point made with facts: cost, size, what it took     | Sans headline, a Mono receipt with the total in cinder |

### Store screenshots

Each screenshot says one practical thing the app does on Windows, in a centred headline of one or two lines, with
what we stand for in the lede (yours, private, free) rather than in the headline. Under it is the proof, large and
running off the bottom edge, drawn after Llama for Windows as it is, in Windows' dark theme: its quick chat with the
real Llama Chat inside, its tray flyout over the taskbar, a close-up of the flyout's models, or one motif or the
app's own highlighted code. The models shown are small and medium ones from llama.app's catalog. The words
keep to the top three quarters, since the Store may lay its own text over the bottom quarter.

## Rules

- One accent per piece. If cinder is on the figure, it is not also on a chip.
- Neutral grounds by default; the day and night skies for the product; sand for a warm, quiet piece.
- Big type and few words. A piece says one thing.
- The real product, the real commands and real numbers, never invented ones presented as real.
- Llama Sans for words, Llama Mono for code and data.
- Keep everything inside the margin and the safe area; nothing touches the edge except a product shot that runs
  off it on purpose.
- No gradients other than the two skies, no glows, no drop shadows on type, no emoji, no stock photography.
