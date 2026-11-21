---
version: alpha
name: Llama UI - Matte Glass
description: >-
  Llama UI, the design system and components of Llama Chat, the chat client llama-server
  bundles. Neutral and quiet: frosted matte surfaces with faint
  blue-grey hairlines, near-flat fills, and status colours for success, warning, error and info. Tokens come in
  three tiers: primitives, semantic tokens (with a "-dark" twin where dark mode differs and a
  "-matte" twin where the Matte light theme does) and components. The colors, typography and
  rounded sections are generated from ui/lib/ui/src/lib/styles by `npm run design:tokens` in
  ui/apps/chat; spacing, components and the prose are kept in sync with the CSS by hand.
colors:
  primary: '{colors.action-primary}'
  brand-ink: '#111111'
  brand-light: '#fafafa'
  neutral-0: 'oklch(1 0 0)'
  neutral-25: 'oklch(0.988 0 0)'
  neutral-50: 'oklch(0.985 0 0)'
  neutral-100: 'oklch(0.97 0 0)'
  neutral-150: 'oklch(0.95 0 0)'
  neutral-200: 'oklch(0.922 0 0)'
  neutral-300: 'oklch(0.87 0 0)'
  neutral-400: 'oklch(0.708 0 0)'
  neutral-500: 'oklch(0.556 0 0)'
  neutral-600: 'oklch(0.439 0 0)'
  neutral-700: 'oklch(0.371 0 0)'
  neutral-750: 'oklch(0.29 0 0)'
  neutral-800: 'oklch(0.269 0 0)'
  neutral-900: 'oklch(0.205 0 0)'
  neutral-925: 'oklch(0.16 0 0)'
  neutral-950: 'oklch(0.145 0 0)'
  sand-50: 'oklch(0.965 0.01 80)'
  sand-100: 'oklch(0.936 0.015 81)'
  sand-200: 'oklch(0.89 0.02 80)'
  sand-400: 'oklch(0.72 0.025 76)'
  sand-900: 'oklch(0.272 0.013 72)'
  art-day-sky: 'oklch(0.78 0.075 245)'
  art-night-sky: 'oklch(0.33 0.07 256)'
  art-cinder: 'oklch(0.68 0.18 46)'
  art-cinder-light: 'oklch(0.85 0.065 46)'
  cool-gray-100: 'oklch(0.938 0.004 250)'
  cool-gray-200: 'oklch(0.91 0.005 250)'
  cool-gray-400: 'oklch(0.81 0.01 250)'
  red-450: 'oklch(0.635 0.229 28)'
  red-550: 'oklch(0.588 0.236 28)'
  red-600: 'oklch(0.55 0.22 27)'
  green-450: 'oklch(0.756 0.197 147)'
  green-550: 'oklch(0.669 0.197 147)'
  green-650: 'oklch(0.52 0.17 147)'
  amber-250: 'oklch(0.842 0.165 84)'
  amber-400: 'oklch(0.752 0.157 78)'
  amber-650: 'oklch(0.547 0.116 74)'
  blue-450: 'oklch(0.67 0.173 256)'
  blue-550: 'oklch(0.587 0.177 256)'
  blue-650: 'oklch(0.535 0.177 256)'
  white-a2: 'oklch(1 0 0 / 0.02)'
  white-a4: 'oklch(1 0 0 / 0.04)'
  white-a6: 'oklch(1 0 0 / 0.06)'
  white-a9: 'oklch(1 0 0 / 0.09)'
  white-a12: 'oklch(1 0 0 / 0.12)'
  white-a18: 'oklch(1 0 0 / 0.18)'
  white-a20: 'oklch(1 0 0 / 0.2)'
  white-a30: 'oklch(1 0 0 / 0.3)'
  white-a50: 'oklch(1 0 0 / 0.5)'
  white-a85: 'oklch(1 0 0 / 0.85)'
  black-a3: 'oklch(0 0 0 / 0.03)'
  black-a4: 'oklch(0 0 0 / 0.04)'
  black-a5: 'oklch(0 0 0 / 0.05)'
  black-a6: 'oklch(0 0 0 / 0.06)'
  black-a8: 'oklch(0 0 0 / 0.08)'
  black-a40: 'oklch(0 0 0 / 0.4)'
  black-a55: 'oklch(0 0 0 / 0.55)'
  surface-page: '{colors.neutral-25}'
  surface-muted: '{colors.neutral-100}'
  surface-emphasis: '{colors.neutral-150}'
  surface-selected: '{colors.neutral-150}'
  surface-raised: '{colors.neutral-0}'
  surface-lifted: '{colors.neutral-0}'
  surface-pressed: '{colors.neutral-0}'
  surface-glass: 'oklch(1 0 0 / 0.8)'
  surface-code: '{colors.neutral-100}'
  surface-overlay: '{colors.black-a40}'
  fg-default: '{colors.neutral-950}'
  fg-soft: 'color-mix(in oklch, {colors.fg-default} 70%, transparent)'
  fg-muted: '{colors.neutral-500}'
  fg-on-action: '{colors.neutral-50}'
  fg-on-danger: '{colors.neutral-0}'
  fg-code: '{colors.neutral-950}'
  fg-danger: '{colors.red-600}'
  fg-success: '{colors.green-650}'
  fg-warning: '{colors.amber-650}'
  fg-info: '{colors.blue-650}'
  line-default: '{colors.neutral-300}'
  line-field: '{colors.neutral-200}'
  line-control: '{colors.neutral-300}'
  line-focus: '{colors.neutral-400}'
  line-danger: '{colors.fill-danger}'
  fill-success: '{colors.green-550}'
  fill-warning: '{colors.amber-400}'
  fill-danger: '{colors.red-550}'
  fill-info: '{colors.blue-550}'
  surface-success: 'color-mix(in oklab, {colors.fill-success} 11%, {colors.surface-page})'
  surface-warning: 'color-mix(in oklab, {colors.fill-warning} 11%, {colors.surface-page})'
  surface-danger: 'color-mix(in oklab, {colors.fill-danger} 11%, {colors.surface-page})'
  surface-info: 'color-mix(in oklab, {colors.fill-info} 11%, {colors.surface-page})'
  line-success-soft: 'color-mix(in oklab, {colors.fill-success} 34%, {colors.line-default})'
  line-warning-soft: 'color-mix(in oklab, {colors.fill-warning} 34%, {colors.line-default})'
  line-danger-soft: 'color-mix(in oklab, {colors.fill-danger} 34%, {colors.line-default})'
  line-info-soft: 'color-mix(in oklab, {colors.fill-info} 34%, {colors.line-default})'
  edge-rest: '{colors.cool-gray-200}'
  edge-focus: '{colors.cool-gray-400}'
  edge-ambient: '{colors.cool-gray-100}'
  edge-raised: '{colors.cool-gray-200}'
  action-primary: '{colors.neutral-900}'
  action-primary-hover: 'color-mix(in oklch, {colors.action-primary} 90%, transparent)'
  action-danger: '{colors.fill-danger}'
  action-danger-hover: 'color-mix(in oklch, {colors.action-danger} 90%, transparent)'
  action-quiet: 'oklch(0.97 0 0 / 0.3)'
  action-quiet-hover: '{colors.neutral-100}'
  action-ghost-hover: 'oklch(0.556 0 0 / 0.1)'
  field-background: '{colors.neutral-0}'
  highlight-inner: '{colors.white-a50}'
  highlight-inverse: '{colors.white-a12}'
  sheen-start: '{colors.white-a20}'
  elevation-ring: '{colors.black-a5}'
  surface-page-matte: '{colors.neutral-0}'
  surface-glass-matte: 'oklch(0.97 0 0 / 0.6)'
  surface-code-matte: '{colors.neutral-50}'
  surface-success-matte: 'color-mix(in oklab, {colors.fill-success} 11%, {colors.surface-page-matte})'
  surface-warning-matte: 'color-mix(in oklab, {colors.fill-warning} 11%, {colors.surface-page-matte})'
  surface-danger-matte: 'color-mix(in oklab, {colors.fill-danger} 11%, {colors.surface-page-matte})'
  surface-info-matte: 'color-mix(in oklab, {colors.fill-info} 11%, {colors.surface-page-matte})'
  field-background-matte: '{colors.surface-page-matte}'
  surface-page-dark: '{colors.neutral-925}'
  surface-muted-dark: '{colors.neutral-800}'
  surface-emphasis-dark: '{colors.neutral-750}'
  surface-selected-dark: '{colors.neutral-800}'
  surface-raised-dark: '{colors.neutral-900}'
  surface-lifted-dark: 'oklch(0.371 0 0 / 0.8)'
  surface-pressed-dark: 'oklch(0.985 0 0 / 0.14)'
  surface-glass-dark: 'oklch(0.269 0 0 / 0.75)'
  surface-code-dark: '{colors.neutral-900}'
  fg-default-dark: '{colors.neutral-50}'
  fg-soft-dark: 'color-mix(in oklch, {colors.fg-default-dark} 70%, transparent)'
  fg-muted-dark: '{colors.neutral-400}'
  fg-on-action-dark: '{colors.neutral-900}'
  fg-code-dark: '{colors.neutral-300}'
  fg-danger-dark: '{colors.fill-danger-dark}'
  fg-success-dark: '{colors.fill-success-dark}'
  fg-warning-dark: '{colors.fill-warning-dark}'
  fg-info-dark: '{colors.fill-info-dark}'
  line-default-dark: '{colors.white-a30}'
  line-field-dark: '{colors.white-a30}'
  line-control-dark: '{colors.white-a12}'
  line-focus-dark: '{colors.neutral-500}'
  line-danger-dark: '{colors.fill-danger-dark}'
  fill-success-dark: '{colors.green-450}'
  fill-warning-dark: '{colors.amber-250}'
  fill-danger-dark: '{colors.red-450}'
  fill-info-dark: '{colors.blue-450}'
  surface-success-dark: 'color-mix(in oklab, {colors.fill-success-dark} 16%, {colors.surface-page-dark})'
  surface-warning-dark: 'color-mix(in oklab, {colors.fill-warning-dark} 16%, {colors.surface-page-dark})'
  surface-danger-dark: 'color-mix(in oklab, {colors.fill-danger-dark} 16%, {colors.surface-page-dark})'
  surface-info-dark: 'color-mix(in oklab, {colors.fill-info-dark} 16%, {colors.surface-page-dark})'
  line-success-soft-dark: 'color-mix(in oklab, {colors.fill-success-dark} 38%, {colors.line-default-dark})'
  line-warning-soft-dark: 'color-mix(in oklab, {colors.fill-warning-dark} 38%, {colors.line-default-dark})'
  line-danger-soft-dark: 'color-mix(in oklab, {colors.fill-danger-dark} 38%, {colors.line-default-dark})'
  line-info-soft-dark: 'color-mix(in oklab, {colors.fill-info-dark} 38%, {colors.line-default-dark})'
  edge-rest-dark: '{colors.white-a6}'
  edge-focus-dark: '{colors.white-a18}'
  edge-ambient-dark: '{colors.white-a4}'
  edge-raised-dark: '{colors.white-a9}'
  action-primary-dark: '{colors.neutral-200}'
  action-primary-hover-dark: 'color-mix(in oklch, {colors.action-primary-dark} 90%, transparent)'
  action-danger-dark: 'color-mix(in oklch, {colors.fill-danger-dark} 58%, transparent)'
  action-danger-hover-dark: 'color-mix(in oklch, {colors.fill-danger-dark} 68%, transparent)'
  action-quiet-dark: 'oklch(0.708 0 0 / 0.15)'
  action-quiet-hover-dark: 'oklch(0.708 0 0 / 0.25)'
  action-ghost-hover-dark: 'oklch(0.708 0 0 / 0.1)'
  field-background-dark: '{colors.white-a9}'
  highlight-inner-dark: '{colors.white-a4}'
  sheen-start-dark: '{colors.white-a2}'
  elevation-ring-dark: '{colors.white-a18}'
typography:
  body:
    fontFamily: 'Llama Sans'
    fontSize: 1rem
    fontWeight: 400
    lineHeight: 1.5rem
  body-sm:
    fontFamily: 'Llama Sans'
    fontSize: 0.875rem
    fontWeight: 400
    lineHeight: 1.25rem
  label:
    fontFamily: 'Llama Sans'
    fontSize: 0.875rem
    fontWeight: 500
    lineHeight: 1.25rem
  label-sm:
    fontFamily: 'Llama Sans'
    fontSize: 0.75rem
    fontWeight: 500
    lineHeight: 1rem
  caption:
    fontFamily: 'Llama Sans'
    fontSize: 0.75rem
    fontWeight: 400
    lineHeight: 1rem
  caption-sm:
    fontFamily: 'Llama Sans'
    fontSize: 0.625rem
    fontWeight: 400
    lineHeight: 0.875rem
  title:
    fontFamily: 'Llama Sans'
    fontSize: 1.125rem
    fontWeight: 600
    lineHeight: 1.75rem
  heading:
    fontFamily: 'Llama Sans'
    fontSize: 1.5rem
    fontWeight: 600
    lineHeight: 2rem
  display:
    fontFamily: 'Llama Sans'
    fontSize: 1.875rem
    fontWeight: 600
    lineHeight: 2.25rem
  code:
    fontFamily: 'Llama Mono'
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
  chat-form:
    backgroundColor: '#f9f9f9'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.3xl}'
    typography: '{typography.body}'
  chat-form-dark:
    backgroundColor: '#202020'
    textColor: '{colors.fg-default-dark}'
  chat-form-placeholder:
    textColor: '{colors.fg-muted}'
  chat-form-placeholder-dark:
    textColor: '{colors.fg-muted-dark}'
  chat-form-edge:
    backgroundColor: '{colors.edge-rest}'
  chat-form-edge-dark:
    backgroundColor: '{colors.edge-rest-dark}'
  chat-form-edge-focus:
    backgroundColor: '{colors.edge-focus}'
  chat-form-edge-focus-dark:
    backgroundColor: '{colors.edge-focus-dark}'
  card:
    backgroundColor: '#f9f9f9'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.xl}'
    padding: '24px'
  card-dark:
    backgroundColor: '#202020'
    textColor: '{colors.fg-default-dark}'
  card-edge:
    backgroundColor: '{colors.edge-rest}'
  card-edge-dark:
    backgroundColor: '{colors.edge-rest-dark}'
  sidebar:
    backgroundColor: '#f9f9f9'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.2xl}'
  sidebar-dark:
    backgroundColor: '#1c1c1c'
    textColor: '{colors.fg-default-dark}'
  sidebar-edge:
    backgroundColor: '{colors.edge-ambient}'
  sidebar-edge-dark:
    backgroundColor: '{colors.edge-ambient-dark}'
  inset-group:
    backgroundColor: 'oklch(1 0 0 / 0.85)'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.xl}'
  inset-group-dark:
    backgroundColor: 'oklch(0.985 0 0 / 0.03)'
    textColor: '{colors.fg-default-dark}'
  inset-group-edge:
    backgroundColor: '{colors.edge-ambient}'
  inset-group-edge-dark:
    backgroundColor: '{colors.edge-ambient-dark}'
  selected-row:
    backgroundColor: 'oklch(0.145 0 0 / 0.05)'
    textColor: '{colors.fg-default}'
  selected-row-dark:
    backgroundColor: 'oklch(0.985 0 0 / 0.05)'
    textColor: '{colors.fg-default-dark}'
  user-bubble:
    backgroundColor: '#f3f3f3'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.2xl}'
    typography: '{typography.body}'
  user-bubble-dark:
    backgroundColor: '#181818'
    textColor: '{colors.fg-default-dark}'
  user-bubble-edge:
    backgroundColor: '{colors.edge-ambient}'
  user-bubble-edge-dark:
    backgroundColor: '{colors.edge-ambient-dark}'
  floating-panel:
    backgroundColor: '#ffffff'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.lg}'
    typography: '{typography.body-sm}'
  floating-panel-dark:
    backgroundColor: '#161616'
    textColor: '{colors.fg-default-dark}'
  floating-panel-edge:
    backgroundColor: '{colors.edge-raised}'
  floating-panel-edge-dark:
    backgroundColor: '{colors.edge-raised-dark}'
  floating-panel-rim:
    backgroundColor: '{colors.highlight-inner}'
  floating-panel-rim-dark:
    backgroundColor: '{colors.highlight-inner-dark}'
  dialog:
    backgroundColor: '#ffffff'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.lg}'
  dialog-dark:
    backgroundColor: '#0d0d0d'
    textColor: '{colors.fg-default-dark}'
  tooltip:
    backgroundColor: '{colors.action-primary}'
    textColor: '{colors.fg-on-action}'
    rounded: '{rounded.md}'
    typography: '{typography.caption}'
  tooltip-dark:
    backgroundColor: '{colors.action-primary-dark}'
    textColor: '{colors.fg-on-action-dark}'
  button-primary:
    backgroundColor: '{colors.action-primary}'
    textColor: '{colors.fg-on-action}'
    rounded: '{rounded.md}'
    typography: '{typography.label}'
    height: '36px'
  button-primary-dark:
    backgroundColor: '{colors.action-primary-dark}'
    textColor: '{colors.fg-on-action-dark}'
  button-primary-hover:
    backgroundColor: '#2e2e2e'
    textColor: '{colors.fg-on-action}'
  button-primary-hover-dark:
    backgroundColor: '#cfcfcf'
    textColor: '{colors.fg-on-action-dark}'
  button-secondary:
    backgroundColor: '#f9f9f9'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.md}'
    typography: '{typography.label}'
    height: '36px'
  button-secondary-dark:
    backgroundColor: '#202020'
    textColor: '{colors.fg-default-dark}'
  button-secondary-hover:
    backgroundColor: '#f7f7f7'
    textColor: '{colors.fg-default}'
  button-secondary-hover-dark:
    backgroundColor: '{colors.surface-muted-dark}'
    textColor: '{colors.fg-default-dark}'
  button-secondary-border:
    backgroundColor: 'oklch(0.87 0 0 / 0.3)'
  button-secondary-border-dark:
    backgroundColor: 'oklch(1 0 0 / 0.06)'
  button-tertiary:
    backgroundColor: '#fcfcfc'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.md}'
    typography: '{typography.label}'
    height: '36px'
  button-tertiary-dark:
    backgroundColor: '#232323'
    textColor: '{colors.fg-default-dark}'
  button-tertiary-hover:
    backgroundColor: '{colors.action-quiet-hover}'
    textColor: '{colors.fg-default}'
  button-tertiary-hover-dark:
    backgroundColor: '#323232'
    textColor: '{colors.fg-default-dark}'
  button-tertiary-border:
    backgroundColor: '{colors.surface-muted}'
  button-tertiary-border-dark:
    backgroundColor: '{colors.surface-muted-dark}'
  button-outline:
    backgroundColor: '{colors.field-background}'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.md}'
    typography: '{typography.label}'
    height: '36px'
  button-outline-dark:
    backgroundColor: '#232323'
    textColor: '{colors.fg-default-dark}'
  button-outline-border:
    backgroundColor: '{colors.line-control}'
  button-outline-border-dark:
    backgroundColor: '{colors.line-control-dark}'
  button-outline-hover:
    backgroundColor: '#f1f1f1'
    textColor: '{colors.fg-default}'
  button-outline-hover-dark:
    backgroundColor: '#1c1c1c'
    textColor: '{colors.fg-default-dark}'
  button-ghost:
    textColor: '{colors.fg-default}'
    rounded: '{rounded.md}'
    typography: '{typography.label}'
  button-ghost-dark:
    textColor: '{colors.fg-default-dark}'
  button-ghost-hover:
    backgroundColor: '#f1f1f1'
    textColor: '{colors.fg-default}'
  button-ghost-hover-dark:
    backgroundColor: '#1c1c1c'
    textColor: '{colors.fg-default-dark}'
  button-destructive:
    backgroundColor: '{colors.action-danger}'
    textColor: '{colors.fg-on-danger}'
    rounded: '{rounded.md}'
    typography: '{typography.label}'
    height: '36px'
  button-destructive-dark:
    backgroundColor: '#952420'
    textColor: '{colors.fg-on-danger}'
  button-destructive-hover:
    backgroundColor: '#ea262d'
    textColor: '{colors.fg-on-danger}'
  button-destructive-hover-dark:
    backgroundColor: '#ac2824'
    textColor: '{colors.fg-on-danger}'
  toggle-on:
    backgroundColor: '{colors.surface-selected}'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.md}'
    typography: '{typography.label}'
    height: '36px'
  toggle-on-dark:
    backgroundColor: '{colors.surface-selected-dark}'
    textColor: '{colors.fg-default-dark}'
  toggle-hover:
    backgroundColor: '#f1f1f1'
    textColor: '{colors.fg-default}'
  toggle-hover-dark:
    backgroundColor: '#1c1c1c'
    textColor: '{colors.fg-default-dark}'
  toggle-group:
    backgroundColor: '#f9f9f9'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.md}'
    typography: '{typography.label}'
    height: '32px'
  toggle-group-dark:
    backgroundColor: '#202020'
    textColor: '{colors.fg-default-dark}'
  toggle-group-border:
    backgroundColor: 'oklch(0.87 0 0 / 0.3)'
  toggle-group-border-dark:
    backgroundColor: 'oklch(1 0 0 / 0.06)'
  toggle-group-seam:
    backgroundColor: '{colors.line-control}'
  toggle-group-seam-dark:
    backgroundColor: '{colors.line-control-dark}'
  toggle-group-key-muted:
    backgroundColor: '{colors.surface-muted}'
    textColor: 'oklch(0.145 0 0 / 0.7)'
  toggle-group-key-muted-dark:
    backgroundColor: '{colors.surface-muted-dark}'
    textColor: 'oklch(0.985 0 0 / 0.7)'
  toggle-group-key-on:
    backgroundColor: '{colors.surface-pressed}'
    textColor: '{colors.fg-default}'
  toggle-group-key-on-dark:
    backgroundColor: '#2e2e2e'
    textColor: '{colors.fg-default-dark}'
  tabs-list:
    backgroundColor: '{colors.surface-muted}'
    rounded: '{rounded.lg}'
    height: '36px'
    padding: '2px'
  tabs-list-dark:
    backgroundColor: '{colors.surface-muted-dark}'
  tabs-trigger:
    backgroundColor: '{colors.surface-muted}'
    textColor: 'oklch(0.145 0 0 / 0.7)'
    rounded: '{rounded.md}'
    typography: '{typography.label}'
  tabs-trigger-dark:
    backgroundColor: '{colors.surface-muted-dark}'
    textColor: 'oklch(0.985 0 0 / 0.7)'
  tabs-trigger-active:
    backgroundColor: '{colors.surface-lifted}'
    textColor: '{colors.fg-default}'
  tabs-trigger-active-dark:
    backgroundColor: '#363636'
    textColor: '{colors.fg-default-dark}'
  switch-track:
    backgroundColor: '{colors.surface-muted}'
    rounded: '{rounded.full}'
    height: '18px'
    width: '32px'
  switch-track-dark:
    backgroundColor: '{colors.surface-muted-dark}'
  switch-track-on:
    backgroundColor: '{colors.action-primary}'
  switch-track-on-dark:
    backgroundColor: '{colors.action-primary-dark}'
  switch-thumb:
    backgroundColor: '{colors.surface-lifted}'
    rounded: '{rounded.full}'
    size: '16px'
  switch-thumb-dark:
    backgroundColor: '{colors.surface-lifted-dark}'
  switch-thumb-on:
    backgroundColor: '{colors.surface-lifted}'
  switch-thumb-on-dark:
    backgroundColor: '{colors.surface-raised-dark}'
  copy-feedback:
    backgroundColor: '{colors.surface-page}'
    textColor: '{colors.fg-success}'
  copy-feedback-dark:
    backgroundColor: '{colors.surface-page-dark}'
    textColor: '{colors.fg-success-dark}'
  input:
    backgroundColor: '{colors.field-background}'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.md}'
    typography: '{typography.body-sm}'
    height: '36px'
  input-dark:
    backgroundColor: '#232323'
    textColor: '{colors.fg-default-dark}'
  input-placeholder:
    textColor: '{colors.fg-muted}'
  input-placeholder-dark:
    textColor: '{colors.fg-muted-dark}'
  input-border:
    backgroundColor: '{colors.line-field}'
  input-border-dark:
    backgroundColor: '{colors.line-field-dark}'
  input-border-focus:
    backgroundColor: '{colors.line-focus}'
  input-border-focus-dark:
    backgroundColor: '{colors.line-focus-dark}'
  input-border-invalid:
    backgroundColor: '{colors.line-danger}'
  input-border-invalid-dark:
    backgroundColor: '{colors.line-danger-dark}'
  badge-default:
    backgroundColor: '{colors.action-primary}'
    textColor: '{colors.fg-on-action}'
    rounded: '{rounded.md}'
    typography: '{typography.label-sm}'
    padding: '2px 8px'
  badge-default-dark:
    backgroundColor: '{colors.action-primary-dark}'
    textColor: '{colors.fg-on-action-dark}'
  badge-secondary:
    backgroundColor: '#dadada'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.md}'
    typography: '{typography.label-sm}'
  badge-secondary-dark:
    backgroundColor: '#252525'
    textColor: '{colors.fg-default-dark}'
  badge-outline:
    textColor: '{colors.fg-default}'
    rounded: '{rounded.md}'
    typography: '{typography.label-sm}'
  badge-outline-dark:
    textColor: '{colors.fg-default-dark}'
  badge-outline-border:
    backgroundColor: '{colors.line-default}'
  badge-outline-border-dark:
    backgroundColor: '{colors.line-default-dark}'
  badge-destructive:
    backgroundColor: '{colors.action-danger}'
    textColor: '{colors.fg-on-danger}'
    rounded: '{rounded.md}'
    typography: '{typography.label-sm}'
  badge-destructive-dark:
    backgroundColor: '#952420'
    textColor: '{colors.fg-on-danger}'
  badge-success:
    backgroundColor: '{colors.surface-success}'
    textColor: '{colors.fg-success}'
    rounded: '{rounded.md}'
    typography: '{typography.label-sm}'
  badge-success-dark:
    backgroundColor: '{colors.surface-success-dark}'
    textColor: '{colors.fg-success-dark}'
  badge-warning:
    backgroundColor: '{colors.surface-warning}'
    textColor: '{colors.fg-warning}'
    rounded: '{rounded.md}'
    typography: '{typography.label-sm}'
  badge-warning-dark:
    backgroundColor: '{colors.surface-warning-dark}'
    textColor: '{colors.fg-warning-dark}'
  badge-danger:
    backgroundColor: '{colors.surface-danger}'
    textColor: '{colors.fg-danger}'
    rounded: '{rounded.md}'
    typography: '{typography.label-sm}'
  badge-danger-dark:
    backgroundColor: '{colors.surface-danger-dark}'
    textColor: '{colors.fg-danger-dark}'
  badge-info:
    backgroundColor: '{colors.surface-info}'
    textColor: '{colors.fg-info}'
    rounded: '{rounded.md}'
    typography: '{typography.label-sm}'
  badge-info-dark:
    backgroundColor: '{colors.surface-info-dark}'
    textColor: '{colors.fg-info-dark}'
  code-block:
    backgroundColor: '{colors.surface-code}'
    textColor: '{colors.fg-code}'
    rounded: '{rounded.lg}'
    typography: '{typography.code}'
  code-block-dark:
    backgroundColor: '{colors.surface-code-dark}'
    textColor: '{colors.fg-code-dark}'
  dialog-overlay:
    backgroundColor: '{colors.surface-overlay}'
  list-row-selected:
    backgroundColor: '{colors.surface-selected}'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.md}'
  list-row-selected-dark:
    backgroundColor: '{colors.surface-selected-dark}'
    textColor: '{colors.fg-default-dark}'
  panel-muted:
    backgroundColor: '{colors.surface-muted}'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.lg}'
  panel-muted-dark:
    backgroundColor: '{colors.surface-muted-dark}'
    textColor: '{colors.fg-default-dark}'
  chip-emphasis:
    backgroundColor: '{colors.surface-emphasis}'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.md}'
  chip-emphasis-dark:
    backgroundColor: '{colors.surface-emphasis-dark}'
    textColor: '{colors.fg-default-dark}'
  page:
    backgroundColor: '{colors.surface-page}'
    textColor: '{colors.fg-default}'
  page-dark:
    backgroundColor: '{colors.surface-page-dark}'
    textColor: '{colors.fg-default-dark}'
  glass-highlight:
    backgroundColor: '{colors.highlight-inner}'
  glass-highlight-dark:
    backgroundColor: '{colors.highlight-inner-dark}'
  tooltip-highlight:
    backgroundColor: '{colors.highlight-inverse}'
  user-bubble-sheen:
    backgroundColor: '{colors.sheen-start}'
  user-bubble-sheen-dark:
    backgroundColor: '{colors.sheen-start-dark}'
  status-lamp-danger:
    backgroundColor: '{colors.fill-danger}'
    rounded: '{rounded.full}'
    size: '8px'
  status-lamp-warning:
    backgroundColor: '{colors.fill-warning}'
    rounded: '{rounded.full}'
    size: '8px'
  status-lamp-success:
    backgroundColor: '{colors.fill-success}'
    rounded: '{rounded.full}'
    size: '8px'
---

## In short

The rules most changes need. The sections below hold the detail and the reasons, and they
win where this list is too short. Before adding a token, a size or a component, use what is
here; if something really is missing, add it here first.

- **Components:** take base components from Llama UI (`@ggml-org/llama-ui/ui/<name>`); do not
  build a new one when an existing one fits. Style them through their variants
  (`variant="outline"`, `size="sm"`), not by overriding classes.
- **Colour:** greys build the UI; colour is for meaning only (status, destructive actions,
  focus). Neutral colours go by shadcn's utility names (`bg-background`, `bg-muted`, `bg-card`,
  `text-foreground`, `text-muted-foreground`, `border-border`), status by `bg-surface-*`,
  `bg-fill-*`, `text-fg-*` and `border-line-*-soft` of `success`, `warning`, `danger` and
  `info`, and any other semantic token as `bg-(--surface-lifted)`. In CSS, read the semantic
  token (`var(--surface-page)`). No Tailwind palette classes, no hex.
- **Type:** Llama Sans, and Llama Mono for code, through roles: `text-body` (16px) for chat
  messages and the chat input, `text-body-sm` (14px) for most text, `text-label` (14px, medium)
  for buttons and labels, `text-label-sm` and `text-caption` (12px) for badges and hints,
  `text-caption-sm` (10px) for tags and counts, `text-title` (18px) and `text-heading` (24px)
  for the rare heading, `text-display` (30px) for the biggest headline. Never Tailwind's own sizes (`text-sm`, `text-xs`) or a size in
  brackets.
- **Shape:** one corner scale, `rounded-sm` to `rounded-4xl`: `rounded-md` (8px) for controls,
  fields and badges, `rounded-lg` (10px) for menus and dialogs, `rounded-xl` (14px) for cards,
  `rounded-2xl` (16px) and `rounded-3xl` (24px) for the larger surfaces, `rounded-full` for
  pills, avatars and lamps.
- **Sizes:** Tailwind's 4px spacing scale. Icons are Lucide, `size-icon-md` (16px), or
  `size-icon-sm` (14px) in compact controls. Controls are `h-control-sm`, `h-control-md` or
  `h-control-lg` (32, 36, 40px).
- **Surfaces:** one surface utility per element (`card-surface`, `matte-float`, `matte-key`,
  `matte-well`), never raw `shadow-*`, `border-*` and `backdrop-blur-*` combinations.
- **Tokens:** read a semantic token directly. A component token is only for a value the
  component decides itself. A value that follows from another token is mixed from it, not
  picked again.
- **No arbitrary values** (`text-[11px]`, `rounded-[1.125rem]`, `p-[7px]`, `z-[20]`): pick the
  nearest from the scale (`z-20`, `max-w-4/5`, `min-w-32`). Brackets are for what no scale can
  hold: a `calc()`, a grid template, a list of transition properties.
- **Motion:** `ease-out-quint` for what enters or answers a press, `ease-sheet` for a sheet
  or a page that slides.
- **Phones:** every dialog is full screen; only alerts stay small and centred.
- **The conversation is the loudest thing on screen;** everything around it stays quiet.

## Overview

**Llama UI** is the design system and component library this file describes. **Llama Chat**,
built from it, is the local, private chat client that llama-server bundles, and the native
Llama apps (Llama for Mac, Llama for Windows) open it in their own window. It should feel calm
and tool-like: neutral greys, plenty of space, and surfaces that read as frosted matte glass
rather than shiny plastic. Depth comes from blur, a faint hairline edge and soft shadows,
never from strong gradients. Colour is reserved for meaning: the status colours (success,
warning, error and info), destructive actions, and the content itself.

There are three layers between a person and a model: the **model** (the weights, a GGUF
file), the **service** (the server that runs inference: llama.cpp and llama-server) and the
**application** (the harness, the UI and UX). The harness, the agentic loop, is its own headless
package, Llama Harness (`@ggml-org/llama-harness`), next to Llama UI (`@ggml-org/llama-ui`). Llama UI and Llama Chat are the application
layer. Running all three locally is the point: people choose their model and keep their
conversations and traces, so the UI never assumes a cloud account or hides which model runs.
The service need not be on the same machine: a server can be run by one person and shared by
many (a family, a team, a community), so the UI should make clear which server a model runs on.

This file is the how, written for agents first. Its principles, as rules to apply here:

- **Function first.** Every element has a job and takes its measurements from a shared role
  (control heights, corners, icon sizes, type roles); never give one component its own.
- **Simple by default, deep on request.** The first screen offers a suggested model and
  sensible defaults; advanced settings (sampling, penalties, system prompts, recipes) sit one
  step away behind a disclosure, never on the default path. Prefer plain words to terms like
  GGUF, quantization or speculative decoding, and explain a term where it must appear.
- **The content is the loudest thing.** The conversation carries the weight and the colour;
  the interface around it stays neutral and quiet.
- **Colour has one job.** Greys build everything; status colours show status and nothing else;
  art colours and cinder are for artwork and the odd brand detail, never for the UI's controls.
- **Recognisable, easy to change.** Keep the character in the icon, the type and one
  accent, so an app built on Llama UI can restyle it through semantic and component tokens alone.
- **Design lives in code.** The CSS is the source of truth; prefer what is generated or
  derived over hand-tuned values, and record the reason next to the value. Content pages are
  Markdown with Svelte components in them, as `.svx` (see `AGENTS.md`).
- **Reuse before adding.** An existing role before a new token, an existing component before
  a new one.

Design tokens come in three tiers, each in its own file under `ui/lib/ui/src/lib/styles/tokens/`:

1. **Primitives** (`primitives.css`): raw colours named by what they are, such as
   `neutral-900`, `cool-gray-200`, `red-550` or `white-a30`. Only semantic tokens point at
   them. Sizes are not primitives: a role holds its own value, and spacing is Tailwind's 4px
   grid.
2. **Semantic tokens** (`semantic.css`): named by purpose, such as `surface-glass`,
   `fg-muted`, `line-field`, `edge-focus`, `action-primary` or `fill-warning`, and the only
   tier that changes between light and dark. Each is either a pick, which names a primitive
   per theme, or a derivation, which is mixed from another semantic token by a fixed rule and
   so follows it through every theme with a single declaration:
   - soft text is the text colour at 70% (`fg-soft`);
   - a hover fill is the fill at 90% (`action-primary-hover`);
   - a status surface is its fill mixed into the page, and a status line its fill mixed into
     the default line, by a share the theme sets (11% and 34% in light, 16% and 38% in dark);
   - a dark shadow is the light one, `elevation-strength` (1 in light, 5 in dark) times as
     dense;
   - the middle of a matte edge is its side at two thirds, and a lamp's rim and glyph are
     its fill mixed 30% towards white and towards black (both in `surfaces.css`).

   The type, icon size and control height roles live here too and are Tailwind utilities:
   `text-label`, `size-icon-md`, `h-control-md`. Corners are Tailwind's own `rounded-sm` to
   `rounded-4xl`, computed from `--radius` in `theme.css`. shadcn's
   colour names (`background`, `muted-foreground`, `border` ...) exist as Tailwind utilities
   only, each mapped to a semantic token in `theme.css`.

3. **Component tokens** (`components.css`): only what a component decides itself, named
   `[component]-[variant]-[property]-[state]`, such as `button-secondary-background-hover` or
   `float-panel-background-modal`: a value no semantic token has, or one that only this
   component changes between themes. A component that takes a semantic token as it is reads
   the semantic token, as `bg-(--action-primary)` and similar, and gets no alias here.

The front matter's `colors`, `typography` and `rounded` are generated from the primitives, the
semantic tokens and the font stacks by `npm run design:tokens` in `ui/apps/chat`, and CI fails
when they are out of date (`npm run check:design`): change the CSS and run it, never edit them
here. The `spacing` and `components` sections and the prose are written by hand; when they
disagree with the CSS, the CSS wins and this file is updated to match. Storybook shows every
tier under **Llama UI › Foundations › Colors**.

Paper light mode is the base. A semantic token that resolves differently in dark mode has a
`-dark` twin here, and one that resolves differently in Matte a `-matte` twin; components list
a `-dark` entry wherever their colours change. Translucent component
fills are given as they render over the page, so contrast checks see the real colour.

## Colors

The palette is achromatic. Hue appears only in the hairline edges (a cool grey at hue 250,
kept almost neutral) and in the status colours, which serve function only: green for success,
amber for warning, red for errors and blue for information, saturated enough to read at a
glance and tuned to text contrast in both modes. Each status has one root colour, its fill (`fill-*`),
and everything in that status derives from it: the soft surface and line of alerts and badges,
the border of an invalid field, the lamp. Red keeps to the same rule, so there is one red: the
danger button and the solid danger badge are the fill itself (`action-danger`), solid in light
mode and frosted, translucent matte glass in dark (58% of the fill over the page). Never style a
danger control with its own red (`bg-destructive`, `bg-red-500` …): use the `destructive`
variant, which `AlertDialog.Action` takes too, or the status tokens. Cinder (`art-cinder`, `oklch(0.68 0.18 46)`, #ee6c1e) is the brand's one accent orange: the .cpp of the llama.cpp
logotype, the mark on a detail in the brand pages and the warm pop in artwork. It is not part of the UI palette. Sand (`sand-50` to `sand-900`) is the warm neutral for communications
(posters, boards, the site); the UI itself stays on the greys.

Backgrounds stay neutral as a rule, in the UI and in communications alike: greys and sand
behind, colour only where it serves a purpose, functional (status, a selection, focus) or
creative (a shape, a numeral, an illustration). Artwork takes its colours from the greys, sand and
the `art-*` primitives: the day sky and the night sky (llama.app's light and dark hero grounds) and
cinder, the accent orange; never from the status colours, and never red.

Light mode has two surface themes, which share everything else (the palette, buttons, fields,
badges, shadows and dark mode): **Paper**, the default, sets white panels, fields and keys on a
faint grey page (`neutral-25`); **Matte** sets grey glass on a white page. Matte is the
`theme-matte` class on `<html>`, chosen with the Light surfaces setting (Settings → General) or
the Surface switch in Storybook's toolbar, and only its surface tokens differ: `surface-page`,
`surface-glass`, `surface-code`, `field-background` and `float-panel-background-modal`.

- **Surfaces:** `surface-page` for the page, `surface-muted` for quiet panels,
  `surface-emphasis` and `surface-selected` for stronger fills and hover or selected rows,
  `surface-raised` for menus and dialogs, `surface-lifted` for a selection raised on a
  recessed track, `surface-pressed` for a key pressed in, `surface-glass` for the translucent matte fill, `surface-code` for code and
  `surface-overlay` behind dialogs.
- **Foreground:** `fg-default` and `fg-muted` for text and icons, `fg-soft` (the text colour
  at 70%) for the label of an option that is not the picked one, `fg-on-action` and
  `fg-on-danger` on filled buttons, `fg-code` for code, `fg-danger` for errors,
  `fg-success` for confirmations such as a copied check, `fg-warning` for cautions and
  `fg-info` for neutral notices.
- **Lines:** `line-default` for dividers, `line-control` for outlined controls (soft in dark
  mode), `line-field` for inputs, `line-focus` for focus and `line-danger` for invalid fields.
- **Edges:** the hairlines of matte surfaces, darker at the sides than in the middle (the
  token is the side; the surface fades it to two thirds in the middle). `edge-rest` at rest,
  `edge-focus` when focus is inside, `edge-ambient` for large quiet containers whose edge
  never changes and, kept even all round, for the user bubble, and `edge-raised` for floating
  panels.
- **Actions:** `action-primary` (near black in light mode, near white in dark mode) and its
  hover, `action-danger` and its hover, and `action-ghost-hover`.
- **Status fills:** `fill-success`, `fill-warning`, `fill-danger` and `fill-info` are each
  status colour at its brightest, for dots, lamps and other small solid marks. They are
  brighter than the matching `fg-*` text colour, which keeps to a readable lightness (4.5:1 on
  the soft surfaces), and are a little lighter again in dark mode.
- **Status surfaces:** `surface-success`, `surface-warning`, `surface-danger` and
  `surface-info` are soft tints of each status fill for alerts and soft badges, with
  `line-*-soft` hairlines to match. They are mixed in oklab so the hue holds over the page.
  Soft status badges are pills with a small dot in the status fill before the label.
- **Status:** `fill-danger`, `fill-warning`, `fill-success` and `fill-info` are the
  lamp fills for error or stopped, loading or warning, ready or connected, and informational. A lamp adds
  a lighter rim and a darker glyph, both mixed from its fill. Known gap: the glyph on the red lamp
  reaches only about 2:1 and on the green lamp about 3.2:1, below WCAG AA, so the lamp colour
  must never be the only signal; pair it with a label or tooltip.

## Typography

The UI uses Llama Sans for all text, headings included (with the system sans-serif stack as its
fallback), and Llama Mono for code (with the system monospace stack as its fallback). Llama Sans
is one variable font with an optical size axis: the browser sets it from the font-size, so text at
16px gets the UI spacing and kerning, 36px and up the tighter Display setting and text under 16px a
looser Small one, with a blend in between. Nothing needs choosing per element; pin a point only to
show it, with `font-variation-settings: 'opsz' 16` or `36`. Its line box is 1.25 at every size, centred on the
cap height; large headings set `line-height: 1.15` themselves. Llama Mono keeps every character,
its coding ligatures included, in one cell, so code and numbers line up in columns. Type is set through roles that fix size and line height together, and for labels and headings the weight too:

- `body` (16/24): chat messages and the chat input.
- `body-sm` (14/20): most text, fields and lists.
- `label` (14/20, 500): buttons and field labels.
- `label-sm` (12/16, 500): badges and compact labels.
- `caption` (12/16): tooltips, hints and metadata.
- `caption-sm` (10/14): the smallest text, for tags, compact badges and counts.
- `title` (18/28, 600) and `heading` (24/32, 600): rare headings.
- `display` (30/36, 600): the biggest headline, such as the greeting on a new chat.

Body and caption text takes the weight around it, regular unless a parent says otherwise; add
a weight class for anything else (`text-body-sm font-semibold`). The roles are the whole
scale: no `text-sm` or `text-xs`, and no size in brackets. On a phone a row that needs larger
text takes the next role up (`text-body-sm max-md:text-body`).

When a component lets callers override the size, keep an explicit weight class next to the
role (`text-label font-medium`) so a size override does not drop the weight. Placeholders
in the chat form are engraved: muted text with a 1px highlight below it in light mode and a
1px shadow above it in dark mode (the `engraved` utility).

## Layout

Spacing follows a 4px grid. Gaps of 4 to 12px (`xs` to `md`) dominate inside controls and
lists; 16 to 32px (`lg` to `2xl`) separate sections. Controls are 32, 36 or 40px tall
(`h-control-sm`, `h-control-md`, `h-control-lg`) and icons are 12, 14, 16 or 20px
(`size-icon-xs` to `size-icon-lg`). Square boxes use `size-N`, never a matching `h-N w-N`
pair, and icons use the `size-icon-*` roles. Inside buttons, toggles and menu items an icon is
16px (`size-icon-md`) unless a smaller control calls for less. The chat column is centred with
a generous bottom offset so the floating chat form never covers the last message. The desktop
sidebar is 18rem wide by default, can be resized, and collapses to a 3rem strip.

### Safe areas

The page runs under the status bar and over the home indicator (`viewport-fit=cover`), which
an installed iPhone app always shows and a browser tab mostly does not. The insets are
`--safe-area-top` and `--safe-area-bottom` (0 where there is none). Anything pinned to a screen
edge keeps clear of them with the safe-area utilities from `src/lib/styles/safe-area.css`,
never a hand-written `calc()`:

- `pt-safe`, `pb-safe`: padding of just the inset, e.g. a sheet that fills the screen.
- `pt-safe-N`, `pb-safe-N`: a spacing step plus the inset, e.g. `pt-safe-12` for the chat,
  whose messages start 3rem below the top on a phone.
- `pt-(--page-top)`: where a page with a title, like Models, Projects or Library, starts on a
  phone: a row below the corner controls, so the title clears both the menu key and the fade
  under the status bar.
- `top-safe-N`, `bottom-safe-N`: the same for absolutely or fixedly placed controls, e.g.
  `top-safe-2.5` for the expanded chat form's collapse button.
- `top-(--corner-control-top)`, `bottom-(--corner-control-bottom)`,
  `right-(--corner-control-side)`: the phone's corner controls, like the menu key and a page's
  floating key. They follow the insets differently
  in the installed app (see Corners below).

**Corners in the installed app.** The status bar and the home indicator set the screen's top
and bottom apart on their own, so a corner control does not add its usual gap on top of their
insets there. At the top it sits just below the status bar (`--corner-control-top`, the inset
plus 0.25rem). At the bottom the home indicator is a short bar in the middle and the corners
beside it are free, so a bottom corner action ignores its inset instead of stacking above the
indicator. Its gaps from the bottom and the side are always equal, 1.75rem
(`--corner-control-bottom` is defined from `--corner-control-side`), which keeps a 56px key
about concentric with an iPhone's rounded screen corner. Only what spans the bottom edge's
middle, like a sheet's last row, clears the inset with `pb-safe` or `bottom-safe-N`. In a
browser tab the top keeps 1rem and the bottom its 1.75rem beyond the insets.

A control placed inside a padded box still needs its own `top-safe-N`: absolute positions
ignore the padding. Dialogs keep clear of the insets on their own (see Dialogs on phones under
Components). At the top, every route on a phone has the `status-bar-fade` from the root
layout: solid behind the status bar and the row of corner controls under it, fading out down to
`--page-top`, like the fade under the desktop tab bar, so content scrolls away under them
instead of colliding with the clock or the chat's header. A route needs no fade of its own;
what covers the whole screen, like the expanded chat form, hides it, and it fades out as the
page slides aside for the sidebar. Check
pinned content with the insets simulated (the Safe areas page in
Storybook does this), since they are 0 in a desktop browser.

## Elevation & Depth

Depth is built from the matte utilities in `ui/lib/ui/src/lib/styles/surfaces.css`. Pick the one that
matches the role of the surface instead of combining raw shadows.

- **`matte-glass`:** large resting surfaces. A translucent fill with a backdrop blur, an
  inner highlight, a soft drop shadow and a 1px hairline whose sides are slightly darker than
  its middle. The hairline is drawn on a masked `::before` so it never tints the fill. Focus
  inside swaps the hairline to the focus edge; the shadow stays the same. Its colours come
  from slot variables, and `chat-form-surface`, `card-surface`, `sidebar-surface` and
  `inset-group-surface` point those slots at one component's tokens, so each component needs
  a single class.
- **`matte-float`** and **`matte-float-modal`:** menus, popovers, select lists, hover cards
  and drawers, and dialogs over the page colour. An 86% opaque panel with blur, a faint edge,
  a rim and a deeper shadow. The edge is a hairline ring baked into the shadow as its last
  layer (`edge-raised`). The panel's 1px border is its rim, the inner highlight
  (`highlight-inner`) right inside the ring; an inset shadow for the rim would sit inside
  the border instead, and the 1px between the two would draw a double edge. In light mode
  dialogs are solid white instead: over the dimmed page a translucent white turns grey.
- **`matte-key`:** buttons, selects, toggles and tabs. The matte finish: on top of the
  button's own fill it adds a light blur, a fill barely lighter at the top (`--key-sheen`, 12% white in light mode, 2% in dark)
  and a small drop shadow. It never changes the button colour and draws no rim of its own: every
  `matte-key` control has a border, and a rim inside it would double the edge. Nothing marks
  the top edge alone, so the edge looks the same all round, with no gloss.
- **`matte-key-solid`:** primary and danger buttons. The same sheen, fainter
  (`--key-sheen-solid`), the same blur, and an even 1px hairline in a shade of the fill drawn
  inside the button (`--key-edge`), so the width does not change. The danger key stays matte
  glass rather than metal: its sheen is almost none (`--button-danger-sheen`), and in dark mode
  it is a frosted, translucent red (`fill-danger` at 58%) whose edge is a soft tint of the red, never
  a lighter rim.
- **`elevated-control`**, **`elevated-surface`** and **`elevated-raised`:** cards, floating
  bars and thumbnails. An elevation token with a hairline ring baked in as its last layer, so
  the edge melts into the shadow and the shadow deepens in dark mode. The ring is
  `--elevation-ring` (black 5%, white 18% in dark), 1px wide. They
  replace a border plus a shadow; never add a border to them.
- **`matte-well`:** inputs, textareas and checkboxes. A shallow inset shade so
  fields read as recessed.
- **`matte-sheen`** and **`matte-sheen-solid`:** the user message bubble and badges. An
  almost invisible top-to-bottom sheen, fainter on solid fills.
- **`status-lamp`** and **`matte-orb`** with `orb-danger`, `orb-warning` or `orb-success`:
  small status dots and the recording button. A near-flat fill barely lighter at the top and
  an even hairline, with no glow.

The elevation tokens are `elevation-surface`, `elevation-control`, `elevation-raised`,
`elevation-inset` and `elevation-pressed`; a tooltip takes `elevation-surface`. The drop shadows are smooth stacks: a few soft
layers instead of one hard one, the closest the darkest and smallest, each further layer about
twice the offset and blur and fainter, the far ones pulled in with a negative spread, so the
shadow fades out instead of ending in a visible edge. They are the only shadows: Tailwind's
own `shadow-sm` to `shadow-2xl` do not follow the theme and are not used. `elevation-surface`
reaches about 20px, `elevation-raised` about 38px, and `elevation-control` is three tight
layers reaching no further than 4px, so buttons and other controls sit close to the page. Two softer steps sit below them: `elevation-soft` for the chat form,
which floats only a little, and `elevation-faint` for the user message bubble, which barely
lifts. The keys inside the chat form, the add key and the model selector, take `elevation-key`:
a 0.5px hairline all round and a 1px drop, since the control elevation's soft
spread blends into the white form; in dark it is the control elevation. Dark mode deepens the same stacks. Tailwind's own `shadow-xs` to `shadow-2xl` are
built the same way (`tokens/semantic.css`), each reaching about as far as Tailwind's default for
it. Every
button carries `elevation-control` (`--button-shadow`); a ghost button only while hovered. Overlays behind dialogs are `surface-overlay`
with a 2px blur.

## Shapes

Corners are soft and consistent, on one scale that is computed from a single base radius
(`--radius`, 10px, in `theme.css`), so changing the base moves every corner with it: `sm`
(6px) for small marks and chips, `md` (8px) for buttons, fields and badges, `lg` (10px) for
menus and dialogs, `xl` (14px) for cards, `2xl` (16px) for the sidebar and message bubbles,
and `3xl` (24px) for the chat form, rising to `4xl` (32px) for the chat form on small
screens. Status lamps, avatars and the round icon buttons in the chat form are fully round.

## Brand

- **Logo:** the llama.cpp mark, one inline SVG (`Logo`) filled with the current text colour,
  so it follows the theme with no light and dark copies. It is sized by `--size`, 1rem by
  default: 14px in model rows, 16px in the sidebar. Colour it with a text token such as
  `fg-default` or `fg-muted`, never a raw colour.
- **App icons:** the mark in `#111111` on white, or `#fafafa` on `#111111` in dark mode
  (`FAVICON_COLORS`, `THEME_COLORS`). The favicon keeps a 4% margin (`FAVICON_PADDING`), so
  the logo spans 96% of it. The maskable and Apple touch icons are white squares with the
  logo at 67% (the generator's 30% padding); its farthest point reaches 73%, inside the 80%
  maskable safe zone. The PWA icons are white circles with the logo at 57%
  (`make-icons-circular.js`, 85% of the square icon).
- **Splash screens:** white, or `#111111` in dark mode, with the logo centred at 24% of the
  screen's shorter side (`SPLASH_PADDING`).
- The icons and splash screens in `static/` are build output of `npm run build-pwa-assets`,
  drawn from `logo.svg`; change the source or the constants, never the generated files.

## Components

- **Buttons:** six variants.
  - `default`: the solid primary button, including the chat submit button.
  - `secondary`: the matte glass button, the same fill as the chat form with a faint border,
    also used for the chat form's Stop and Skip reasoning buttons.
  - `tertiary`: a faint grey tint with a soft muted border, quieter than secondary and
    still a visible button; kept for low-emphasis actions that need more presence than
    ghost.
  - `outline`: the input fill with a `line-control` border, for secondary actions in dialogs
    and forms.
  - `ghost`: no fill until hovered, for icon buttons and quiet actions.
  - `destructive`: red, for deleting and other irreversible actions.

  All except ghost carry `matte-key`. Sizes are sm (32px), default (36px) and lg (40px), plus
  square icon sizes.

- **Badges:** four variants with the same names: `default`, `secondary` (a light foreground
  tint), `outline` and `destructive`, plus the soft status badges `success`, `warning`, `danger` and
  `info`. Badges are a fill or a border with the faint matte sheen and no inner rim.
- **Toggles:** the same corners, type, heights, padding and icon size as buttons, with
  square `icon-sm`, `icon` and `icon-lg` sizes for icon-only toggles. `default` has no fill
  until hovered; `outline` borrows the outline button. Pressed toggles take
  `surface-selected`.
- **Button groups:** joined buttons share one shadow, drawn by the group, and one border at
  each seam. A group separator replaces the seam border. Filled buttons draw a seam in their
  own text colour at 25% instead.
- **Selection on a track:** a choice that moves along a track is a recessed track with the
  selection raised on it, never a row of outlined buttons. The track is `surface-muted` with
  the inset shade (`elevation-inset`); the selection is a `surface-lifted` chip with
  `elevation-control`, and unselected options are softened text (`fg-soft`) with no fill. Tabs use the
  same track for their list, with the active tab raised; a `line` variant
  underlines the active tab instead. The chip or underline slides between tabs in 300ms on a
  plain ease-out, with a stretch of at most 5% along the move that is gone by the time it
  lands; with reduced motion it jumps instead. The switch is the same track with a raised
  thumb; switched on, the track takes the primary colour. The message statistics view switcher
  uses the tabs tokens.
- **Toggle groups:** a row of keys, like an old stereo's, not a track. At rest the group is
  a secondary button, the same glass fill, border and shadow as a select trigger, with a
  `line-control` seam between keys. A pressed key turns white (`surface-pressed`, the lightest
  fill in dark mode) and sinks in under a subtle inner shadow (`elevation-pressed`), its label
  a pixel lower. While any key is pressed, the others turn grey in light mode
  (`surface-muted`) and mute their labels, so the pressed one stands out. A group is as tall
  as a control of its size.
- **Chat form:** one `chat-form-surface` holding the input, attachments and actions. Its fill
  is the muted grey at 60% in light mode and 75% in dark mode.
- **User message bubble:** the primary colour at 5% with `matte-sheen` and the very faint
  `edge-ambient` hairline, so it separates from the page without a visible border.
- **Sidebar:** `sidebar-surface`, the muted fill at 60% with no inner highlight and the
  ambient edge. On small screens it keeps a plain `shadow-md` instead.
- **Inset groups:** `inset-group-surface` gathers rows inside a glass panel, like the
  sidebar footer's Models, MCP and Settings, instead of a rule between them and the rows
  above. In light mode it is a soft white (`surface-raised` at 85%) on the grey glass; in dark
  mode a raised fill would be darker than the glass, so it is a 3% tint of the text colour. It
  takes the sidebar's ambient edge, no inner highlight and only a faint, wide shade under it
  (`0 2px 8px -4px`, 8% black in light, 40% in dark), with the card radius. It reads as part of
  the panel: no tight contact shadow and no solid border, which make a button, and no inset
  shade, which makes a field. Unlike the other `matte-glass` surfaces it does not blur: it
  already sits on glass, and a second backdrop blur re-paints the panel's fill behind it and
  turns the group visibly lighter, whatever its own fill.
- **Selected rows:** `selected-row-surface` marks the row a sidebar list is on: a
  conversation, a project or a top action, on desktop and on a phone. It is a 5% tint of the
  text colour, under hover's 10%, with the inset group's edge and shade, so the selected row
  reads as picked out rather than hovered, in both modes. It keeps the row's own radius, and
  hover and press fills still paint over it.
- **Fields:** `matte-well` with a 1px `line-field` outline and the standard 3px focus ring.
- **Floating panels:** every menu, popover, select list, hover card and drawer uses
  `matte-float`; dialogs use `matte-float-modal`.
- **Chat header on phones:** a chat shows its title and, under it, its working directory
  centred between the menu key and the context gauge, on the top fade. The directory is the
  folder's name (the full path is its title) and tapping it opens the directory picker, which
  on a phone is a drawer. A new chat has no header yet: it offers "Set working directory" in
  the + menu instead, and no chip sits under the form on a phone. The header steps aside with
  the corner keys while the expanded chat form covers the screen.
- **Dialogs on phones:** below `md` every dialog is full screen, whatever size it takes on a
  desktop. The shared `Dialog.Content` does this, so a dialog sets only its desktop size and
  never its own phone size. Its top and bottom borders take the height of the status bar and
  the home indicator, transparent, so the content keeps clear of both with no change to its
  padding (see Safe areas). A dialog that draws its own full-screen content instead, like the
  attachment preview, keeps clear with `pt-safe`, `pb-safe` and `top-safe-N` itself. Alerts
  (`AlertDialog`: confirmations, renames, errors) are the exception: a short question with a
  button or two, they stay small and centred on a phone too. A place the sidebar opens, like
  Settings, MCP Servers or Servers, is no dialog on a phone at all but a page at its own route
  (`#/settings`, `#/mcp-servers`, `#/servers`), with the menu key and a title like any phone page; a
  desktop or tablet keeps it a dialog over the page it was opened from.
- **Glass keys:** a round key that floats over the page and lets it show through, like the
  menu and context gauge keys at a phone's top corners and the key back to a chat's bottom, is
  a `GlassKey` (`GLASS_KEY_CLASS`): white glass (`surface-glass`), a faint `line-default`
  hairline, and `matte-key`'s blur and saturation with its small shadow, so what scrolls under
  it softens instead of showing sharp behind the icon. 40px at the corners, 36px over the
  page; its icon is a 2px-stroke glyph in `fg-default` on the 24px icon grid, the context
  gauge's dial drawn the same way. Don't rebuild one from a ghost `Button` plus classes: the
  button's own blur overrides `matte-key`'s.
- **A page's main action on phones:** below `md` the action that adds to a page, like New
  project or New prompt, leaves the page's header for a `FloatingActionButton`: a round 56px
  `default` (primary) key with a large shadow and only a `+` icon, pinned in the bottom-right
  corner 1.75rem from both edges (`--corner-control-side`, `--corner-control-bottom`; in the
  installed app beside the home indicator rather than above it, see Safe areas), where the
  thumb rests. It is portalled to the body, so its place on screen never depends on the page's
  box, and slides with the page when the sidebar opens, by the same offset and ease
  (`--sidebar-reveal-x`), taking no taps while the page is aside. The page keeps room under its
  last row (`max-md:pb-safe-24`) so the key never covers it. One per page, only for its main
  action; from `md` up the action stays a button in the header.
- **Voice clips:** a recorded clip is a round primary play key, its waveform and the time in
  `caption`: bars in the muted foreground at 40%, the played part in the primary colour, so
  the clip stays neutral in both modes. The live waveform while recording is as tall as the
  input's line (24px), the newest level on the right, and holds still with reduced motion. A
  voice message's transcript opens under the clip on hover, keyboard focus or a tap, and
  stays in the DOM for screen readers while closed.
- **Copy feedback:** a copy action turns its icon into a `fg-success` check for 1.5s
  (`COPY_FEEDBACK_RESET_MS`), only when the copy succeeded, then back. The icon pops in with a
  150ms fade and scale, skipped on first render and with reduced motion.
- **Tooltips:** the inverse of the page, the primary colour with a faint inner rim.
- **Status:** server, model and MCP connection states use the status lamps, never coloured
  text alone.

## Do's and Don'ts

- Do keep fills flat. Gradients, if any, must be barely perceptible; a visible gradient
  makes the UI look like a 2000s interface.
- Do keep edges faint. At rest a hairline should blend in; on focus it should be noticeable
  but never strong.
- Do keep the shadow the same on focus. Focus is shown by the edge and the focus ring, not by
  lifting the surface.
- Do use the status colours only for status. Buttons keep their neutral colours, and the
  submit button stays black in light mode and white in dark mode.
- Do read a semantic token directly wherever a component takes it as it is. Add a component
  token only for a value the component decides itself, never as an alias of a semantic token.
- Do add a new colour as a primitive first, then give it a semantic name before any
  component uses it.
- Do give motion a job: show where something went or what changed, like a selection sliding
  to its new place. A barely visible stretch along a move, or a settle past the target by a
  pixel or two, can make that feel physical; keep it under about 5% and 300ms, and drop it
  entirely with reduced motion. Never let a bounce carry something past the edge of its
  container, such as a tab indicator past its track; there, slide without one. If you notice
  the effect itself rather than the change it shows, it is too much.
- Do reach for the surface utility that matches the component instead of adding raw
  `shadow-*`, `border-*` and `backdrop-blur-*` combinations.
- Don't use Tailwind's palette classes (`text-amber-500`, `bg-green-500/10`) for meaning in
  new code; use the semantic tokens. Some older components still do and are being moved over.
- Don't introduce new accent hues (such as a blue submit button) without a design decision.
- Don't put white rims or highlights on large ambient containers such as the sidebar.
- Don't animate for decoration: no motion on things that did not change, no bounces on
  entrances, and nothing that delays the user.
- Don't stack two glass surfaces with blur directly on top of each other.
- Don't give a dialog its own size on a phone: every dialog is full screen there. Only alerts
  stay small and centred.
- Don't put more than one floating key on a phone page, or use one for anything but the
  page's main action.
- Don't lift a bottom corner action above the home indicator's inset in the installed app,
  or give it a different gap from the bottom than from the side: the corners beside the
  indicator are free, and `--corner-control-bottom` and `--corner-control-side` keep the two
  gaps equal.
