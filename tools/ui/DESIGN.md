---
version: alpha
name: llama.cpp Web UI - Matte Glass
description: >-
  Neutral, quiet chat interface for llama-server. Frosted matte surfaces with faint
  blue-grey hairlines, near-flat fills, and red/yellow/green status lamps. Light mode is
  the base token set; every token used in dark mode has a "-dark" twin.
colors:
  primary: 'oklch(0.205 0 0)'
  on-primary: 'oklch(0.985 0 0)'
  secondary: 'oklch(0.95 0 0)'
  on-secondary: 'oklch(0.205 0 0)'
  surface: 'oklch(1 0 0)'
  on-surface: 'oklch(0.145 0 0)'
  surface-muted: 'oklch(0.97 0 0)'
  on-surface-muted: 'oklch(0.556 0 0)'
  surface-raised: 'oklch(1 0 0)'
  accent: 'oklch(0.95 0 0)'
  border: 'oklch(0.875 0 0)'
  input: 'oklch(0.92 0 0)'
  ring: 'oklch(0.708 0 0)'
  error: 'oklch(0.577 0.245 27.325)'
  on-error: 'oklch(0.985 0 0)'
  code-surface: 'oklch(0.985 0 0)'
  on-code-surface: 'oklch(0.145 0 0)'
  glass-fill: 'oklch(0.97 0 0 / 0.6)'
  glass-edge: 'oklch(0.91 0.005 250)'
  glass-edge-focus: 'oklch(0.81 0.01 250)'
  ambient-edge: 'oklch(0.938 0.004 250)'
  bubble-fill: '#f3f3f3'
  bubble-edge: 'oklch(0.935 0.003 250)'
  float-edge: 'oklch(0.9 0.005 250)'
  status-red: 'oklch(0.6 0.235 27)'
  status-red-ink: 'oklch(0.43 0.17 27)'
  status-yellow: 'oklch(0.85 0.17 84)'
  status-yellow-ink: 'oklch(0.5 0.1 70)'
  status-green: 'oklch(0.73 0.2 148)'
  status-green-ink: 'oklch(0.45 0.14 145)'
  primary-dark: 'oklch(0.922 0 0)'
  on-primary-dark: 'oklch(0.205 0 0)'
  secondary-dark: 'oklch(0.29 0 0)'
  on-secondary-dark: 'oklch(0.985 0 0)'
  surface-dark: 'oklch(0.16 0 0)'
  on-surface-dark: 'oklch(0.985 0 0)'
  surface-muted-dark: 'oklch(0.269 0 0)'
  on-surface-muted-dark: 'oklch(0.708 0 0)'
  surface-raised-dark: 'oklch(0.205 0 0)'
  accent-dark: 'oklch(0.269 0 0)'
  border-dark: 'oklch(1 0 0 / 30%)'
  input-dark: 'oklch(1 0 0 / 30%)'
  ring-dark: 'oklch(0.556 0 0)'
  error-dark: 'oklch(0.704 0.191 22.216)'
  error-fill-dark: '#9e4143'
  code-surface-dark: 'oklch(0.225 0 0)'
  on-code-surface-dark: 'oklch(0.875 0 0)'
  glass-fill-dark: 'oklch(0.269 0 0 / 0.75)'
  glass-edge-dark: 'oklch(1 0 0 / 0.06)'
  glass-edge-focus-dark: 'oklch(1 0 0 / 0.18)'
  ambient-edge-dark: 'oklch(1 0 0 / 0.042)'
  bubble-fill-dark: '#181818'
  bubble-edge-dark: 'oklch(1 0 0 / 0.035)'
  float-edge-dark: 'oklch(1 0 0 / 0.09)'
  status-red-dark: 'oklch(0.63 0.22 25)'
  status-red-ink-dark: 'oklch(0.4 0.15 27)'
  status-yellow-dark: 'oklch(0.83 0.16 84)'
  status-yellow-ink-dark: 'oklch(0.45 0.09 70)'
  status-green-dark: 'oklch(0.72 0.19 148)'
  status-green-ink-dark: 'oklch(0.4 0.12 145)'
typography:
  headline-lg:
    fontFamily: system-ui
    fontSize: 1.5rem
    fontWeight: 600
    lineHeight: 2rem
  headline-md:
    fontFamily: system-ui
    fontSize: 1.125rem
    fontWeight: 600
    lineHeight: 1.75rem
  body-md:
    fontFamily: system-ui
    fontSize: 1rem
    fontWeight: 400
    lineHeight: 1.5rem
  body-sm:
    fontFamily: system-ui
    fontSize: 0.875rem
    fontWeight: 400
    lineHeight: 1.25rem
  label-md:
    fontFamily: system-ui
    fontSize: 0.875rem
    fontWeight: 500
    lineHeight: 1.25rem
  label-sm:
    fontFamily: system-ui
    fontSize: 0.75rem
    fontWeight: 500
    lineHeight: 1rem
  code:
    fontFamily: ui-monospace
    fontSize: 0.875rem
    fontWeight: 400
    lineHeight: 1.25rem
rounded:
  sm: 6px
  md: 8px
  lg: 10px
  xl: 14px
  2xl: 16px
  3xl: 24px
  4xl: 32px
  full: 9999px
spacing:
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 24px
  2xl: 32px
components:
  button-primary:
    backgroundColor: '{colors.primary}'
    textColor: '{colors.on-primary}'
    typography: '{typography.label-md}'
    rounded: '{rounded.md}'
    height: 36px
    padding: 0 16px
  button-primary-dark:
    backgroundColor: '{colors.primary-dark}'
    textColor: '{colors.on-primary-dark}'
  button-secondary:
    backgroundColor: '{colors.secondary}'
    textColor: '{colors.on-secondary}'
    typography: '{typography.label-md}'
    rounded: '{rounded.md}'
    height: 36px
  button-secondary-dark:
    backgroundColor: '{colors.secondary-dark}'
    textColor: '{colors.on-secondary-dark}'
  button-ghost-hover:
    backgroundColor: '{colors.accent}'
    textColor: '{colors.on-surface}'
  button-ghost-hover-dark:
    backgroundColor: '{colors.accent-dark}'
    textColor: '{colors.on-surface-dark}'
  button-destructive:
    backgroundColor: '{colors.error}'
    textColor: '{colors.on-error}'
    rounded: '{rounded.md}'
  button-destructive-dark:
    backgroundColor: '{colors.error-fill-dark}'
    textColor: '{colors.on-error}'
  input-invalid-dark:
    backgroundColor: '{colors.error-dark}'
  input:
    backgroundColor: '{colors.surface}'
    textColor: '{colors.on-surface}'
    typography: '{typography.body-sm}'
    rounded: '{rounded.md}'
    height: 36px
  input-border:
    backgroundColor: '{colors.input}'
  input-focus-ring:
    backgroundColor: '{colors.ring}'
  input-dark:
    backgroundColor: '{colors.surface-dark}'
    textColor: '{colors.on-surface-dark}'
  input-border-dark:
    backgroundColor: '{colors.input-dark}'
  input-focus-ring-dark:
    backgroundColor: '{colors.ring-dark}'
  chat-form:
    backgroundColor: '{colors.glass-fill}'
    textColor: '{colors.on-surface}'
    typography: '{typography.body-md}'
    rounded: '{rounded.3xl}'
  chat-form-edge:
    backgroundColor: '{colors.glass-edge}'
  chat-form-edge-focus:
    backgroundColor: '{colors.glass-edge-focus}'
  chat-form-placeholder:
    textColor: '{colors.on-surface-muted}'
  chat-form-dark:
    backgroundColor: '{colors.glass-fill-dark}'
    textColor: '{colors.on-surface-dark}'
  chat-form-edge-dark:
    backgroundColor: '{colors.glass-edge-dark}'
  chat-form-edge-focus-dark:
    backgroundColor: '{colors.glass-edge-focus-dark}'
  chat-form-placeholder-dark:
    textColor: '{colors.on-surface-muted-dark}'
  sidebar:
    backgroundColor: '{colors.surface-muted}'
    textColor: '{colors.on-surface}'
    rounded: '{rounded.2xl}'
  sidebar-edge:
    backgroundColor: '{colors.ambient-edge}'
  sidebar-dark:
    backgroundColor: '{colors.surface-muted-dark}'
    textColor: '{colors.on-surface-dark}'
  sidebar-edge-dark:
    backgroundColor: '{colors.ambient-edge-dark}'
  user-bubble:
    backgroundColor: '{colors.bubble-fill}'
    textColor: '{colors.on-surface}'
    typography: '{typography.body-md}'
    rounded: '{rounded.3xl}'
  user-bubble-edge:
    backgroundColor: '{colors.bubble-edge}'
  user-bubble-dark:
    backgroundColor: '{colors.bubble-fill-dark}'
    textColor: '{colors.on-surface-dark}'
  user-bubble-edge-dark:
    backgroundColor: '{colors.bubble-edge-dark}'
  floating-panel:
    backgroundColor: '{colors.surface-raised}'
    textColor: '{colors.on-surface}'
    typography: '{typography.body-sm}'
    rounded: '{rounded.lg}'
  floating-panel-edge:
    backgroundColor: '{colors.float-edge}'
  floating-panel-dark:
    backgroundColor: '{colors.surface-raised-dark}'
    textColor: '{colors.on-surface-dark}'
  floating-panel-edge-dark:
    backgroundColor: '{colors.float-edge-dark}'
  card:
    backgroundColor: '{colors.glass-fill}'
    textColor: '{colors.on-surface}'
    rounded: '{rounded.xl}'
    padding: 24px
  card-dark:
    backgroundColor: '{colors.glass-fill-dark}'
    textColor: '{colors.on-surface-dark}'
  badge:
    backgroundColor: '{colors.secondary}'
    textColor: '{colors.on-secondary}'
    typography: '{typography.label-sm}'
    rounded: '{rounded.md}'
    padding: 2px 8px
  badge-outline-edge:
    backgroundColor: '{colors.border}'
  badge-dark:
    backgroundColor: '{colors.secondary-dark}'
    textColor: '{colors.on-secondary-dark}'
  badge-outline-edge-dark:
    backgroundColor: '{colors.border-dark}'
  code-block:
    backgroundColor: '{colors.code-surface}'
    textColor: '{colors.on-code-surface}'
    typography: '{typography.code}'
    rounded: '{rounded.lg}'
  code-block-dark:
    backgroundColor: '{colors.code-surface-dark}'
    textColor: '{colors.on-code-surface-dark}'
  status-lamp-red:
    backgroundColor: '{colors.status-red}'
    textColor: '{colors.status-red-ink}'
    rounded: '{rounded.full}'
    size: 8px
  status-lamp-yellow:
    backgroundColor: '{colors.status-yellow}'
    textColor: '{colors.status-yellow-ink}'
    rounded: '{rounded.full}'
    size: 8px
  status-lamp-green:
    backgroundColor: '{colors.status-green}'
    textColor: '{colors.status-green-ink}'
    rounded: '{rounded.full}'
    size: 8px
  status-lamp-red-dark:
    backgroundColor: '{colors.status-red-dark}'
    textColor: '{colors.status-red-ink-dark}'
  status-lamp-yellow-dark:
    backgroundColor: '{colors.status-yellow-dark}'
    textColor: '{colors.status-yellow-ink-dark}'
  status-lamp-green-dark:
    backgroundColor: '{colors.status-green-dark}'
    textColor: '{colors.status-green-ink-dark}'
---

## Overview

The llama.cpp web UI is a local, private chat client for llama-server. It should feel calm
and tool-like: neutral greys, plenty of space, and surfaces that read as frosted matte glass
rather than shiny plastic. Depth comes from blur, a faint hairline edge and soft shadows,
never from strong gradients. Colour is reserved for meaning: the red, yellow and green
status lamps, destructive actions, and the content itself.

This file describes intent. The implementation lives in `src/app.css` (tokens on `:root`
and `.dark`, plus the `matte-*` utilities) and is shown in Storybook under
**Design system**. When the two disagree, `src/app.css` wins and this file should be updated.

Light mode is the base token set. Dark mode uses the token of the same name with a `-dark`
suffix, and the components below list a `-dark` entry wherever their colours change.

## Colors

The palette is achromatic. Hue appears only in the hairline edges (a cool blue-grey at
hue 250, kept almost neutral) and in status and error colours.

- **Primary (`primary`):** near black in light mode, near white in dark mode. Used for the
  default button, including the chat submit button, and for strong text. Do not tint it.
- **Secondary and accent (`secondary`, `accent`):** soft grey fills for secondary buttons,
  hover states and selected rows.
- **Surface (`surface`, `surface-muted`, `surface-raised`):** the page, muted panels such
  as the sidebar, and raised panels such as menus and dialogs.
- **Text (`on-surface`, `on-surface-muted`):** body text and secondary text, hints and
  placeholders.
- **Lines (`border`, `input`, `ring`):** dividers, field outlines and the focus ring.
- **Error (`error`):** destructive buttons and invalid fields only. In dark mode the
  destructive button uses `error-dark` at 60% over the page (`error-fill-dark`) with white
  text, while invalid field outlines use `error-dark` itself.
- **Glass (`glass-fill`, `glass-edge`, `glass-edge-focus`):** the translucent fill and
  hairline of matte glass surfaces. The edge darkens on focus in light mode and lightens in
  dark mode.
- **Ambient edge (`ambient-edge`):** an even fainter hairline for large, quiet containers
  such as the sidebar. It does not change on focus.
- **Status (`status-red`, `status-yellow`, `status-green`):** lamp fills for error or
  stopped, loading or warning, and ready or connected. Each has an `-ink` shade for glyphs
  and text on top of the lamp. Known gap: the ink on the red lamp reaches only 2:1 to 2.6:1
  and on the green lamp about 3.2:1 to 3.8:1, below WCAG AA, so the lamp colour must never be
  the only signal; pair it with a label or tooltip.

## Typography

The UI uses the system sans-serif stack (`system-ui`) and the system monospace stack for
code. Most of the interface is small: `body-sm` (14px) and `label-sm` (12px) cover nearly
all controls, lists and metadata. `body-md` (16px) is for chat messages and the chat input.
Headlines are rare and stay at `headline-md` or `headline-lg`; there is no display size.
Use at most two weights on a screen, normally 400 and 500, with 600 reserved for headings.

Placeholders in the chat form are engraved: muted text with a 1px highlight below it in
light mode and a 1px shadow above it in dark mode (the `engraved` utility).

## Layout

Spacing follows a 4px grid. Gaps of 4 to 12px (`xs` to `md`) dominate inside controls and
lists; 16 to 32px (`lg` to `2xl`) separate sections. The chat column is centred with a
generous bottom offset so the floating chat form never covers the last message. The desktop
sidebar is 18rem wide by default, can be resized, and collapses to a 3rem strip.

## Elevation & Depth

Depth is built from the matte utilities in `src/app.css`. Pick the one that matches the
role of the surface instead of combining raw shadows.

- **`matte-glass`:** large resting surfaces such as the chat form and cards. A translucent
  fill with a 24px backdrop blur, an inner highlight, a soft drop shadow and a 1px hairline
  whose sides are slightly darker than its middle. The hairline is drawn on a masked
  `::before` so it never tints the translucent fill. Focus inside the surface swaps the
  hairline to the focus edge; the shadow stays the same.
- **`matte-float`:** menus, popovers, select lists, hover cards, dialogs and drawers. An
  86% opaque panel colour with blur, a faint edge and a deeper drop shadow.
- **`matte-key`:** buttons. Adds a light blur, a thin inner rim and a small drop shadow on
  top of the button's own fill. It never changes the button colour.
- **`matte-well`:** inputs, textareas, checkboxes and switches. A shallow inset shade so
  fields read as recessed.
- **`matte-chip`:** badges. A thin inner rim only.
- **`matte-sheen`:** the user message bubble. An almost invisible top-to-bottom sheen.
- **`status-lamp` and `orb-red`, `orb-yellow`, `orb-green`:** small status dots.
  **`matte-orb`** is the larger version used by the recording button.

Overlays behind dialogs are black at 40% with a 2px blur.

## Shapes

Corners are soft and consistent. Small controls use `md` (8px), panels and menus use `lg`
(10px), cards use `xl` (14px), the sidebar uses `2xl` (16px), and the chat form and message
bubbles use `3xl` (24px), rising to `4xl` (32px) for the chat form on small screens. Status
lamps, avatars and icon buttons in the chat form are fully round.

## Components

- **Buttons:** variants are default (primary), secondary, tertiary, outline, ghost,
  destructive and link. All except ghost and link carry `matte-key`. Sizes are sm (32px),
  default (36px) and lg (40px), plus square icon sizes.
- **Chat form:** one `matte-glass` container holding the input, attachments and actions.
  Its fill is muted grey at 60% in light mode and 75% in dark mode. The submit button is
  the plain primary button.
- **User message bubble:** the primary colour at 5% (`bg-primary/5`) with `matte-sheen` and
  the very faint `bubble-edge` hairline, so it separates from the page without a visible
  border. `bubble-fill` is that tint as it renders over the page background.
- **Sidebar:** `matte-glass` with the muted fill at 60%, no inner highlight and the
  ambient edge. On small screens it keeps a plain `shadow-md` instead.
- **Fields:** `matte-well` with a 1px `input` outline and the standard 3px focus ring.
- **Floating panels:** every menu, popover, select list, hover card, dialog and drawer uses
  `matte-float`. Dialogs base it on the page background colour.
- **Badges:** model and capability badges use `matte-chip`.
- **Status:** server, model and MCP connection states use the status lamps, never coloured
  text alone.

## Do's and Don'ts

- Do keep fills flat. Gradients, if any, must be barely perceptible; a visible gradient
  makes the UI look like a 2000s interface.
- Do keep edges faint. At rest a hairline should blend in; on focus it should be noticeable
  but never strong.
- Do keep the shadow the same on focus. Focus is shown by the edge and the focus ring, not by
  lifting the surface.
- Do use the status lamp colours only for status. Buttons keep their neutral colours, and
  the submit button stays black in light mode and white in dark mode.
- Do reach for the `matte-*` utility that matches the surface's role instead of adding raw
  `shadow-*`, `border-*` and `backdrop-blur-*` combinations.
- Don't introduce new accent hues (such as a blue submit button) without a design decision.
- Don't put white rims or highlights on large ambient containers such as the sidebar.
- Don't stack two glass surfaces with blur directly on top of each other.
- Don't hardcode colours in components; use the tokens so both modes stay in sync.
