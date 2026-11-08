# Llama Brand and communications

This is the brand book for **Llama** (the app) and **llama.cpp** (the engine): the rules for posts, posters,
release images, video thumbnails, link previews, banners and any other piece the team makes. It is written for
agents first (Claude Design reads it as the README of the Llama design system) and for people too.

**The Storybook is the source of truth.** Every rule here is drawn there with the real fonts, tokens and
components, under **Communications** (Direction, Foundations, Elements, Modules, Key visuals, Applications),
**Brand** (Logo, Typography, Colors) and **Philosophy**. When this file and the Storybook disagree, the Storybook
wins. This file lives at `tools/ui/BRAND.md` in llama.cpp; the Llama design system in Claude is exported from it
and from the Storybook by `npm run export:design-system`. The how of the app's own UI is `DESIGN.md`.

## What we stand for

AI that is personal and private: the model runs on a device you already own (a phone, a laptop, a desktop), your
conversations stay on it, and it works offline, without an account. Open models on the hardware people already
have put it within reach of anyone, and a server one person runs can be shared by many. Nothing comes
pre-installed: the apps make the first choices for people starting out, recommending a model that fits the device,
and leave every setting open to power users. We say all of this by showing it, never by preaching it, and point at
no one. The reasons are in Storybook's **Philosophy**.

### Voice

- Plain words and real numbers. Short sentences. Name things the way people know them.
- To the point and lean, without padding. Lean isn't terse: keep the words that give a line its life.
- Calm and hopeful. No pride, no righteousness, no fear and no hype.
- Show what the app does rather than mock the alternative.
- Name model makers in words, never with their logos, except inside the real app where it shows them (its model
  lists). Never pick favourites among models or vendors.

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
- **Llama Mono** only for what is code or data: commands, URLs and repo ids, figures in tables, and
  small uppercase labels (0.12em tracking). Never for headlines or running text.
- Figures are tabular. A number in a piece is real and states its unit.

Sizes are multiples of the format's unit `u` (see Grid):

| Role     | Size  | Notes                                                                |
| -------- | ----- | -------------------------------------------------------------------- |
| Display  | 3.2u  | Bold, line height 0.92, -0.035em. One statement, two to three lines. |
| Headline | 2u    | Bold, line height 0.98, -0.02em.                                     |
| Figure   | 5u    | Bold. One number that carries the piece.                             |
| Lede     | 0.84u | Regular, muted ink, line height 1.3; larger for phones (see below).  |
| Body     | 0.62u | Regular, line height 1.4.                                            |
| Label    | 0.54u | Llama Mono Medium, uppercase, 0.12em tracking, muted.                |
| Mono     | 0.58u | Llama Mono, commands and data.                                       |
| Footer   | 0.58u | Who it is from and the promise, muted.                               |

The text stack keeps one rhythm in every layout: a label 0.6u over its headline, the headline 1.2u over its lede,
and 1.6u from the lede to what follows it (a command, chips, a caption). Space it by these, not by eye. A lede keeps
to 17u at most (about 40 characters a line), so it never runs up to a motif or diagram beside it.

Pieces are read on phones, at a quarter to a third of their size, so small text is never smaller than these: the
footer, a card's details, a URL and a command at 0.58u or more, a kicker at 0.5u at the least. When a piece runs
out of room, tighten the tracking and leading first, then move the split between text and motif or cut padding;
don't shrink the type.

Type that has to hold more tracks tighter rather than shrinks: a lede set larger than 0.84u (up to 1u) at -0.015em
and line height 1.25, a headline down to -0.035em, Mono at 0.6u and up at -0.02em to -0.03em. Never tighter than
that, and never looser than the defaults.

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
- **Art colours** (`art-day-sky`, `art-night-sky`, `art-cinder-light`) are for
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
| Mac App Store 16:10   | 2880 × 1800 | Mac App Store screenshots: a 1440 × 900 screen at 2x                  | 75    |

## Building blocks

- **Header**: the lockup (or the icon before a Mono kicker), an optional pill with a cinder dot, the URL in Mono at
  the right, at least 2u from the kicker or pill. When the row can't hold that, the URL goes beside the label of what
  it points to (as a model drop's catalog URL beside its cards), or into the footer.
- **Footer**: who it is from on the left, the promise on the right, in muted ink.
- **Chips**: Mono pills. Outline for use cases, solid for a tag, cinder once.
- **Figure**: a big number with a Mono label under it.
- **Command**: a terminal line in Mono with a cinder `$`, on the ground's inset, always on one line. A line of config
  or code goes without the `$`.
- **App icon**: Llama as a thing in a piece (a node in a diagram, an app among apps), from the app icon file: the
  tile as the app ships it, its corners at 22.5% of its side and the icon at 67% of it, optically centred, in the
  ground's ink. Never a square drawn round the icon. About 2.4u, with at least 1u clear round it; its name and
  address go under it, and lines to it start half a unit off its edge.
- **Model card**: maker and purpose, name, a short description, figures in cinder, Mono facts and the repo id. Its
  small lines keep the phone sizes whatever the card's size (details 0.58u, description 0.62u, tags 0.5u); a name
  never breaks (its badge goes under it), and a repo id breaks only after its owner. In a single image the repo id is
  left to the link.
- **Carousel**: on Instagram and LinkedIn (square, portrait, story), a piece with several items, such as a model
  drop, is a cover and then one slide per item, since one image can't hold every item at a size a phone shows. The
  cover has the headline, the lede and what is inside (a row per item: its name and tag); each slide has the same
  header with its place (2 / 3) where the URL would be, and the item's card large.
- **Stat card**: a figure, its label and what it means.
- **Product shot**: the real Llama app (its menu bar and panel), never a mock-up.
- **Motifs**: line drawings in the ground's ink with one cinder detail. Models and maths: decision tree (decision
  models), neural network, tensor, gradient descent, vector space. Charts and fields: curve, quantised blocks, dot
  grid. Machines, each isometric or flat: GPU, laptop, small workstation, large workstation, server. A motif left
  less than 4.5u of height is left out rather than drawn as a speck.

## Key visuals

Layouts built from the blocks on the grid. A new piece starts from one of them.

| Key visual | Use it for                                            | Shape                                            |
| ---------- | ----------------------------------------------------- | ------------------------------------------------ |
| Statement  | One thing said big: a launch line, a manifesto        | Display headline, lede, a motif filling the rest |
| Product    | The app itself: link previews, launch images          | Headline beside the product shot, a caption chip |
| Roster     | Several items of one kind: a model drop               | Headline with a motif, chips, a row of cards     |
| Bulletin   | News with several items: what's new, release notes    | Headline and lede, a lead item and a side column |
| Numeral    | One figure that carries the point: speed, size, count | The figure huge in cinder over a dot grid        |

### Store screenshots

Each screenshot says one practical thing the app does, in a centred headline of one or two lines, with what we stand
for in the lede (yours, private, free) rather than in the headline. Under it is the proof, large, drawn after the app
as it is (Llama for Windows in Windows' dark theme): its quick chat with the real Llama Chat inside, the quick chat
over another app with its shortcut's keys, its tray flyout over the taskbar, its Models card up close (the list, or a
model's page with its fit verdicts), or one figure: a motif, a receipt, a number, the app's own highlighted code. Only
what the app already does, from its release notes and README. The models shown are small and medium ones from
llama.app's catalog. The words keep to the top three quarters, since the Store may lay its own text over the bottom
quarter. The opener and the chat come in several versions, to compare or to A/B test.

The Mac App Store's set is the same voice at 16:10, drawn after Llama for Mac in light mode: its menu (rebuilt from
the app's source) dropped from the menu bar, Llama Chat in a Safari web app's window, and the menu's QR code beside a
phone that scanned it. It claims nothing about telemetry: the Mac app reports crashes.

## Rules

- One accent per piece. If cinder is on the figure, it is not also on a chip.
- Neutral grounds by default; the day and night skies for the product; sand for a warm, quiet piece.
- Big type, to the point. A piece says one thing. When it doesn't fit, tighten the tracking before the type size.
- The real product, the real commands and real numbers, never invented ones presented as real.
- Llama Sans for words, Llama Mono for code and data.
- Keep everything inside the margin and the safe area; nothing touches the edge except a product shot that runs
  off it on purpose.
- Separate runs of text on one line keep at least 2u apart (items of one list, such as a card's figures, at least
  1u), so they never read as one line, and nothing runs into the header or footer. When they can't, move one of them
  to its own place rather than squeezing them together.
- No gradients other than the two skies, no glows, no drop shadows on type, no emoji, no stock photography.
