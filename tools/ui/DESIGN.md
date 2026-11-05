---
version: alpha
name: llama.cpp Web UI - Matte Glass
description: >-
  Neutral, quiet chat interface for llama-server. Frosted matte surfaces with faint
  blue-grey hairlines, near-flat fills, and status colours for success, warning, error and info. Tokens come in
  three tiers: primitives, semantic tokens (with a "-dark" twin where dark mode differs)
  and components. Generated from src/lib/styles/tokens/*.css.
colors:
  primary: '{colors.action-primary}'
  neutral-0: 'oklch(1 0 0)'
  neutral-50: 'oklch(0.985 0 0)'
  neutral-100: 'oklch(0.97 0 0)'
  neutral-150: 'oklch(0.95 0 0)'
  neutral-200: 'oklch(0.922 0 0)'
  neutral-300: 'oklch(0.87 0 0)'
  neutral-400: 'oklch(0.708 0 0)'
  neutral-500: 'oklch(0.556 0 0)'
  neutral-750: 'oklch(0.29 0 0)'
  neutral-800: 'oklch(0.269 0 0)'
  neutral-900: 'oklch(0.205 0 0)'
  neutral-925: 'oklch(0.16 0 0)'
  neutral-950: 'oklch(0.145 0 0)'
  cool-gray-50: 'oklch(0.958 0.002 250)'
  cool-gray-100: 'oklch(0.938 0.004 250)'
  cool-gray-200: 'oklch(0.91 0.005 250)'
  cool-gray-300: 'oklch(0.86 0.007 250)'
  cool-gray-400: 'oklch(0.81 0.01 250)'
  red-400: 'oklch(0.69 0.2 27)'
  red-500: 'oklch(0.6 0.235 27)'
  red-450: 'oklch(0.635 0.229 28)'
  red-550: 'oklch(0.588 0.236 28)'
  red-600: 'oklch(0.55 0.22 27)'
  red-800: 'oklch(0.43 0.17 27)'
  yellow-300: 'oklch(0.9 0.14 84)'
  yellow-400: 'oklch(0.85 0.17 84)'
  yellow-800: 'oklch(0.5 0.1 84)'
  green-400: 'oklch(0.81 0.16 148)'
  green-450: 'oklch(0.756 0.197 147)'
  green-500: 'oklch(0.73 0.2 148)'
  green-550: 'oklch(0.669 0.197 147)'
  green-650: 'oklch(0.52 0.17 147)'
  green-800: 'oklch(0.45 0.14 148)'
  amber-250: 'oklch(0.842 0.165 84)'
  amber-400: 'oklch(0.752 0.157 78)'
  amber-650: 'oklch(0.547 0.116 74)'
  blue-400: 'oklch(0.7 0.163 255)'
  blue-450: 'oklch(0.67 0.173 256)'
  blue-500: 'oklch(0.62 0.15 255)'
  blue-550: 'oklch(0.587 0.177 256)'
  blue-650: 'oklch(0.535 0.177 256)'
  blue-800: 'oklch(0.42 0.11 255)'
  white-a2: 'oklch(1 0 0 / 0.02)'
  white-a4: 'oklch(1 0 0 / 0.04)'
  white-a6: 'oklch(1 0 0 / 0.06)'
  white-a9: 'oklch(1 0 0 / 0.09)'
  white-a12: 'oklch(1 0 0 / 0.12)'
  white-a18: 'oklch(1 0 0 / 0.18)'
  white-a20: 'oklch(1 0 0 / 0.2)'
  white-a30: 'oklch(1 0 0 / 0.3)'
  white-a50: 'oklch(1 0 0 / 0.5)'
  black-a6: 'oklch(0 0 0 / 0.06)'
  black-a30: 'oklch(0 0 0 / 0.3)'
  black-a40: 'oklch(0 0 0 / 0.4)'
  surface-page: '{colors.neutral-0}'
  surface-muted: '{colors.neutral-100}'
  surface-emphasis: '{colors.neutral-150}'
  surface-selected: '{colors.neutral-150}'
  surface-raised: '{colors.neutral-0}'
  surface-lifted: '{colors.neutral-0}'
  surface-pressed: '{colors.neutral-0}'
  surface-glass: 'oklch(0.97 0 0 / 0.6)'
  surface-code: '{colors.neutral-50}'
  surface-overlay: '{colors.black-a40}'
  fg-default: '{colors.neutral-950}'
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
  line-danger: '{colors.red-600}'
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
  edge-rest-side: '{colors.cool-gray-200}'
  edge-rest-mid: '{colors.cool-gray-100}'
  edge-focus-side: '{colors.cool-gray-400}'
  edge-focus-mid: '{colors.cool-gray-300}'
  edge-ambient-side: '{colors.cool-gray-100}'
  edge-ambient-mid: '{colors.cool-gray-50}'
  edge-faint: '{colors.cool-gray-100}'
  edge-raised: '{colors.cool-gray-200}'
  action-primary: '{colors.neutral-900}'
  action-primary-hover: 'oklch(0.205 0 0 / 0.9)'
  action-danger: '{colors.red-600}'
  action-danger-hover: 'oklch(0.55 0.22 27 / 0.9)'
  action-quiet: 'oklch(0.97 0 0 / 0.3)'
  action-quiet-hover: '{colors.neutral-100}'
  action-ghost-hover: 'oklch(0.556 0 0 / 0.1)'
  field-background: 'oklch(1 0 0)'
  field-shade: '{colors.black-a6}'
  status-danger: '{colors.fill-danger}'
  status-danger-rim: '{colors.red-400}'
  status-danger-glyph: '{colors.red-800}'
  status-warning: '{colors.fill-warning}'
  status-warning-rim: '{colors.yellow-300}'
  status-warning-glyph: '{colors.yellow-800}'
  status-success: '{colors.fill-success}'
  status-success-rim: '{colors.green-400}'
  status-success-glyph: '{colors.green-800}'
  status-info: '{colors.fill-info}'
  status-info-rim: '{colors.blue-400}'
  status-info-glyph: '{colors.blue-800}'
  highlight-inner: '{colors.white-a50}'
  highlight-inverse: '{colors.white-a12}'
  sheen-start: '{colors.white-a20}'
  sheen-end: 'transparent'
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
  fg-muted-dark: '{colors.neutral-400}'
  fg-on-action-dark: '{colors.neutral-900}'
  fg-code-dark: '{colors.neutral-300}'
  fg-danger-dark: '{colors.red-400}'
  fg-success-dark: '{colors.green-450}'
  fg-warning-dark: '{colors.amber-250}'
  fg-info-dark: '{colors.blue-400}'
  line-default-dark: '{colors.white-a30}'
  line-field-dark: '{colors.white-a30}'
  line-control-dark: '{colors.white-a12}'
  line-focus-dark: '{colors.neutral-500}'
  line-danger-dark: '{colors.red-400}'
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
  edge-rest-side-dark: '{colors.white-a6}'
  edge-rest-mid-dark: '{colors.white-a4}'
  edge-focus-side-dark: '{colors.white-a18}'
  edge-focus-mid-dark: '{colors.white-a12}'
  edge-ambient-side-dark: '{colors.white-a4}'
  edge-ambient-mid-dark: '{colors.white-a2}'
  edge-faint-dark: '{colors.white-a4}'
  edge-raised-dark: '{colors.white-a9}'
  action-primary-dark: '{colors.neutral-200}'
  action-primary-hover-dark: 'oklch(0.922 0 0 / 0.9)'
  action-danger-dark: 'oklch(0.69 0.2 27 / 0.6)'
  action-danger-hover-dark: 'oklch(0.69 0.2 27 / 0.6)'
  action-quiet-dark: 'oklch(0.708 0 0 / 0.15)'
  action-quiet-hover-dark: 'oklch(0.708 0 0 / 0.25)'
  action-ghost-hover-dark: 'oklch(0.708 0 0 / 0.1)'
  field-background-dark: '{colors.white-a9}'
  field-shade-dark: '{colors.black-a30}'
  highlight-inner-dark: '{colors.white-a4}'
  sheen-start-dark: '{colors.white-a2}'
typography:
  heading:
    fontFamily: system-ui
    fontSize: 1.5rem
    fontWeight: 600
    lineHeight: 2rem
  title:
    fontFamily: system-ui
    fontSize: 1.125rem
    fontWeight: 600
    lineHeight: 1.75rem
  body:
    fontFamily: system-ui
    fontSize: 1rem
    fontWeight: 400
    lineHeight: 1.5rem
  body-sm:
    fontFamily: system-ui
    fontSize: 0.875rem
    fontWeight: 400
    lineHeight: 1.25rem
  label:
    fontFamily: system-ui
    fontSize: 0.875rem
    fontWeight: 500
    lineHeight: 1.25rem
  label-sm:
    fontFamily: system-ui
    fontSize: 0.75rem
    fontWeight: 500
    lineHeight: 1rem
  caption:
    fontFamily: system-ui
    fontSize: 0.75rem
    fontWeight: 400
    lineHeight: 1rem
  code:
    fontFamily: ui-monospace
    fontSize: 0.875rem
    fontWeight: 400
    lineHeight: 1.25rem
rounded:
  control: 0.5rem
  panel: 0.625rem
  card: 0.875rem
  container: 1rem
  surface: 1.5rem
  surface-lg: 2rem
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
    rounded: '{rounded.surface}'
    typography: '{typography.body}'
  chat-form-dark:
    backgroundColor: '#202020'
    textColor: '{colors.fg-default-dark}'
  chat-form-placeholder:
    textColor: '{colors.fg-muted}'
  chat-form-placeholder-dark:
    textColor: '{colors.fg-muted-dark}'
  chat-form-edge:
    backgroundColor: '{colors.edge-rest-side}'
  chat-form-edge-dark:
    backgroundColor: '{colors.edge-rest-side-dark}'
  chat-form-edge-focus:
    backgroundColor: '{colors.edge-focus-side}'
  chat-form-edge-focus-dark:
    backgroundColor: '{colors.edge-focus-side-dark}'
  card:
    backgroundColor: '#f9f9f9'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.card}'
    padding: '24px'
  card-dark:
    backgroundColor: '#202020'
    textColor: '{colors.fg-default-dark}'
  card-edge:
    backgroundColor: '{colors.edge-rest-side}'
  card-edge-dark:
    backgroundColor: '{colors.edge-rest-side-dark}'
  sidebar:
    backgroundColor: '#f9f9f9'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.container}'
  sidebar-dark:
    backgroundColor: '#1c1c1c'
    textColor: '{colors.fg-default-dark}'
  sidebar-edge:
    backgroundColor: '{colors.edge-ambient-side}'
  sidebar-edge-dark:
    backgroundColor: '{colors.edge-ambient-side-dark}'
  user-bubble:
    backgroundColor: '#f3f3f3'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.surface}'
    typography: '{typography.body}'
  user-bubble-dark:
    backgroundColor: '#181818'
    textColor: '{colors.fg-default-dark}'
  user-bubble-edge:
    backgroundColor: '{colors.edge-faint}'
  user-bubble-edge-dark:
    backgroundColor: '{colors.edge-faint-dark}'
  floating-panel:
    backgroundColor: '#ffffff'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.panel}'
    typography: '{typography.body-sm}'
  floating-panel-dark:
    backgroundColor: '#161616'
    textColor: '{colors.fg-default-dark}'
  floating-panel-edge:
    backgroundColor: '{colors.edge-raised}'
  floating-panel-edge-dark:
    backgroundColor: '{colors.edge-raised-dark}'
  dialog:
    backgroundColor: '#ffffff'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.panel}'
  dialog-dark:
    backgroundColor: '#0d0d0d'
    textColor: '{colors.fg-default-dark}'
  tooltip:
    backgroundColor: '{colors.action-primary}'
    textColor: '{colors.fg-on-action}'
    rounded: '{rounded.control}'
    typography: '{typography.caption}'
  tooltip-dark:
    backgroundColor: '{colors.action-primary-dark}'
    textColor: '{colors.fg-on-action-dark}'
  button-primary:
    backgroundColor: '{colors.action-primary}'
    textColor: '{colors.fg-on-action}'
    rounded: '{rounded.control}'
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
    rounded: '{rounded.control}'
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
    rounded: '{rounded.control}'
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
    rounded: '{rounded.control}'
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
    rounded: '{rounded.control}'
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
    rounded: '{rounded.control}'
    typography: '{typography.label}'
    height: '36px'
  button-destructive-dark:
    backgroundColor: '#9e3d37'
    textColor: '{colors.fg-on-danger}'
  button-destructive-hover:
    backgroundColor: '#d82431'
    textColor: '{colors.fg-on-danger}'
  button-destructive-hover-dark:
    backgroundColor: '#9e3d37'
    textColor: '{colors.fg-on-danger}'
  toggle-on:
    backgroundColor: '{colors.surface-selected}'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.control}'
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
    rounded: '{rounded.control}'
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
    rounded: '{rounded.panel}'
    height: '36px'
    padding: '2px'
  tabs-list-dark:
    backgroundColor: '{colors.surface-muted-dark}'
  tabs-trigger:
    backgroundColor: '{colors.surface-muted}'
    textColor: 'oklch(0.145 0 0 / 0.7)'
    rounded: '{rounded.control}'
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
    rounded: '{rounded.control}'
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
    rounded: '{rounded.control}'
    typography: '{typography.label-sm}'
    padding: '2px 8px'
  badge-default-dark:
    backgroundColor: '{colors.action-primary-dark}'
    textColor: '{colors.fg-on-action-dark}'
  badge-secondary:
    backgroundColor: '#dadada'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.control}'
    typography: '{typography.label-sm}'
  badge-secondary-dark:
    backgroundColor: '#252525'
    textColor: '{colors.fg-default-dark}'
  badge-outline:
    textColor: '{colors.fg-default}'
    rounded: '{rounded.control}'
    typography: '{typography.label-sm}'
  badge-outline-dark:
    textColor: '{colors.fg-default-dark}'
  badge-outline-border:
    backgroundColor: '{colors.line-default}'
  badge-outline-border-dark:
    backgroundColor: '{colors.line-default-dark}'
  badge-destructive:
    backgroundColor: '{colors.fg-danger}'
    textColor: '{colors.fg-on-danger}'
    rounded: '{rounded.control}'
    typography: '{typography.label-sm}'
  badge-destructive-dark:
    backgroundColor: '#b6443e'
    textColor: '{colors.fg-on-danger}'
  badge-success:
    backgroundColor: '{colors.surface-success}'
    textColor: '{colors.fg-success}'
    rounded: '{rounded.control}'
    typography: '{typography.label-sm}'
  badge-success-dark:
    backgroundColor: '{colors.surface-success-dark}'
    textColor: '{colors.fg-success-dark}'
  badge-warning:
    backgroundColor: '{colors.surface-warning}'
    textColor: '{colors.fg-warning}'
    rounded: '{rounded.control}'
    typography: '{typography.label-sm}'
  badge-warning-dark:
    backgroundColor: '{colors.surface-warning-dark}'
    textColor: '{colors.fg-warning-dark}'
  badge-danger:
    backgroundColor: '{colors.surface-danger}'
    textColor: '{colors.fg-danger}'
    rounded: '{rounded.control}'
    typography: '{typography.label-sm}'
  badge-danger-dark:
    backgroundColor: '{colors.surface-danger-dark}'
    textColor: '{colors.fg-danger-dark}'
  badge-info:
    backgroundColor: '{colors.surface-info}'
    textColor: '{colors.fg-info}'
    rounded: '{rounded.control}'
    typography: '{typography.label-sm}'
  badge-info-dark:
    backgroundColor: '{colors.surface-info-dark}'
    textColor: '{colors.fg-info-dark}'
  code-block:
    backgroundColor: '{colors.surface-code}'
    textColor: '{colors.fg-code}'
    rounded: '{rounded.panel}'
    typography: '{typography.code}'
  code-block-dark:
    backgroundColor: '{colors.surface-code-dark}'
    textColor: '{colors.fg-code-dark}'
  dialog-overlay:
    backgroundColor: '{colors.surface-overlay}'
  list-row-selected:
    backgroundColor: '{colors.surface-selected}'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.control}'
  list-row-selected-dark:
    backgroundColor: '{colors.surface-selected-dark}'
    textColor: '{colors.fg-default-dark}'
  panel-muted:
    backgroundColor: '{colors.surface-muted}'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.panel}'
  panel-muted-dark:
    backgroundColor: '{colors.surface-muted-dark}'
    textColor: '{colors.fg-default-dark}'
  chip-emphasis:
    backgroundColor: '{colors.surface-emphasis}'
    textColor: '{colors.fg-default}'
    rounded: '{rounded.control}'
  chip-emphasis-dark:
    backgroundColor: '{colors.surface-emphasis-dark}'
    textColor: '{colors.fg-default-dark}'
  page:
    backgroundColor: '{colors.surface-page}'
    textColor: '{colors.fg-default}'
  page-dark:
    backgroundColor: '{colors.surface-page-dark}'
    textColor: '{colors.fg-default-dark}'
  chat-form-edge-mid:
    backgroundColor: '{colors.edge-rest-mid}'
  chat-form-edge-mid-dark:
    backgroundColor: '{colors.edge-rest-mid-dark}'
  chat-form-edge-mid-focus:
    backgroundColor: '{colors.edge-focus-mid}'
  chat-form-edge-mid-focus-dark:
    backgroundColor: '{colors.edge-focus-mid-dark}'
  sidebar-edge-mid:
    backgroundColor: '{colors.edge-ambient-mid}'
  sidebar-edge-mid-dark:
    backgroundColor: '{colors.edge-ambient-mid-dark}'
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
  field-shade:
    backgroundColor: '{colors.field-shade}'
  field-shade-dark:
    backgroundColor: '{colors.field-shade-dark}'
  status-lamp-danger-rim:
    backgroundColor: '{colors.status-danger-rim}'
  status-lamp-warning-rim:
    backgroundColor: '{colors.status-warning-rim}'
  status-lamp-success-rim:
    backgroundColor: '{colors.status-success-rim}'
  status-lamp-danger:
    backgroundColor: '{colors.status-danger}'
    textColor: '{colors.status-danger-glyph}'
    rounded: '{rounded.full}'
    size: '8px'
  status-lamp-warning:
    backgroundColor: '{colors.status-warning}'
    textColor: '{colors.status-warning-glyph}'
    rounded: '{rounded.full}'
    size: '8px'
  status-lamp-success:
    backgroundColor: '{colors.status-success}'
    textColor: '{colors.status-success-glyph}'
    rounded: '{rounded.full}'
    size: '8px'
---

## Overview

The llama.cpp web UI is a local, private chat client for llama-server. It should feel calm
and tool-like: neutral greys, plenty of space, and surfaces that read as frosted matte glass
rather than shiny plastic. Depth comes from blur, a faint hairline edge and soft shadows,
never from strong gradients. Colour is reserved for meaning: the status colours (success,
warning, error and info), destructive actions, and the content itself.

Design tokens come in three tiers, each in its own file under `src/lib/styles/tokens/`:

1. **Primitives** (`primitives.css`): raw values named by what they are, such as
   `neutral-900`, `cool-gray-200`, `red-500`, `white-a30`, `font-size-14` or `length-24`.
   Only semantic tokens point at them.
2. **Semantic tokens** (`semantic.css`): named by purpose, such as `surface-glass`,
   `fg-muted`, `line-field`, `edge-focus-side`, `action-primary` or `status-warning`. They
   point at primitives and are the only tier that changes between light and dark. The type,
   corner, icon size and control height roles live here too and are Tailwind utilities:
   `text-label`, `rounded-control`, `size-icon-md`, `h-control-md`.
3. **Component tokens** (`components.css`): one group per component, named
   `[component]-[variant]-[property]-[state]`, such as `chat-form-edge-side-focus` or
   `button-secondary-background-hover`. They point at semantic tokens, and components read
   them as `bg-(--button-primary-background)` and similar.

The front matter of this file mirrors those three files. When they disagree, the CSS wins
and this file should be updated to match. Storybook shows every tier under **Design
system > Foundations > Colors**.

Light mode is the base. A semantic token whose dark value differs has a `-dark` twin here,
and components list a `-dark` entry wherever their colours change. Translucent component
fills are given as they render over the page, so contrast checks see the real colour.

## Colors

The palette is achromatic. Hue appears only in the hairline edges (a cool grey at hue 250,
kept almost neutral) and in the status colours, which serve function only: green for success,
amber for warning, red for errors and blue for information, saturated enough to read at a
glance and tuned to text contrast in both modes. The brand orange belongs to the llama.cpp
logotype and is not part of the UI palette.

- **Surfaces:** `surface-page` for the page, `surface-muted` for quiet panels,
  `surface-emphasis` and `surface-selected` for stronger fills and hover or selected rows,
  `surface-raised` for menus and dialogs, `surface-lifted` for a selection raised on a
  recessed track, `surface-pressed` for a key pressed in, `surface-glass` for the translucent matte fill, `surface-code` for code and
  `surface-overlay` behind dialogs.
- **Foreground:** `fg-default` and `fg-muted` for text and icons, `fg-on-action` and
  `fg-on-danger` on filled buttons, `fg-code` for code, `fg-danger` for errors,
  `fg-success` for confirmations such as a copied check, `fg-warning` for cautions and
  `fg-info` for neutral notices.
- **Lines:** `line-default` for dividers, `line-control` for outlined controls (soft in dark
  mode), `line-field` for inputs, `line-focus` for focus and `line-danger` for invalid fields.
- **Edges:** the hairlines of matte surfaces, darker at the sides than in the middle.
  `edge-rest-*` at rest, `edge-focus-*` when focus is inside, `edge-ambient-*` for large
  quiet containers whose edge never changes, `edge-faint` for the user bubble and
  `edge-raised` for floating panels.
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
- **Status:** `status-danger`, `status-warning`, `status-success` and `status-info` are the
  lamp fills for error or stopped, loading or warning, ready or connected, and informational. They take the status
  fills, and each has a lighter `-rim` and a darker `-glyph`. Known gap: the glyph on the red lamp
  reaches only about 2:1 and on the green lamp about 3.2:1, below WCAG AA, so the lamp colour
  must never be the only signal; pair it with a label or tooltip.

## Typography

The UI uses Llama Sans UI (with the system sans-serif stack as its fallback) and the system
monospace stack for code. Type is set through roles that fix size, line height and weight together:

- `body` (16/24, 400): chat messages and the chat input.
- `body-sm` (14/20, 400): most text, fields and lists.
- `label` (14/20, 500): buttons and field labels.
- `label-sm` (12/16, 500): badges and compact labels.
- `caption` (12/16, 400): tooltips, hints and metadata.
- `title` (18/28, 600) and `heading` (24/32, 600): rare headings; there is no display size.

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

## Elevation & Depth

Depth is built from the matte utilities in `src/lib/styles/surfaces.css`. Pick the one that
matches the role of the surface instead of combining raw shadows.

- **`matte-glass`:** large resting surfaces. A translucent fill with a backdrop blur, an
  inner highlight, a soft drop shadow and a 1px hairline whose sides are slightly darker than
  its middle. The hairline is drawn on a masked `::before` so it never tints the fill. Focus
  inside swaps the hairline to the focus edge; the shadow stays the same. Its colours come
  from slot variables, and `chat-form-surface`, `card-surface` and `sidebar-surface` point
  those slots at one component's tokens, so each component needs a single class.
- **`matte-float`** and **`matte-float-modal`:** menus, popovers, select lists, hover cards
  and drawers, and dialogs over the page colour. An 86% opaque panel with blur, a faint edge
  and a deeper shadow. The edge is a hairline ring baked into the shadow, as shadow-plugin's
  `smooth-shadow-ring-*` draws it, never a border next to a shadow, which would read as a
  double edge; the panel's border stays transparent so its size does not change. In light mode dialogs are solid white instead: over the dimmed page a
  translucent white turns grey.
- **`matte-key`:** buttons, selects, toggles and tabs. The matte finish: on top of the
  button's own fill it adds a light blur, a fill barely lighter at the top (`--key-sheen`) and
  a small drop shadow. It never changes the button colour and draws no rim of its own: every
  `matte-key` control has a border, and a rim inside it would double the edge. Nothing marks
  the top edge alone, so the edge looks the same all round, with no gloss.
- **`matte-key-solid`:** primary and danger buttons. The same sheen, fainter
  (`--key-sheen-solid`), and an even 1px hairline in a shade of the fill drawn inside the
  button (`--key-edge`), so the width does not change.
- **`elevated-control`**, **`elevated-surface`** and **`elevated-raised`:** cards, floating
  bars and thumbnails. An elevation token with a 1px hairline ring baked in as its last layer,
  like shadow-plugin's `smooth-shadow-ring-*` but on our tokens, so the shadow deepens in dark
  mode. The ring reads `--smooth-ring-color`; tint it with `smooth-ring-*`. They replace a
  border plus a shadow; never add a border to them.
- **`matte-well`:** inputs, textareas and checkboxes. A shallow inset shade so
  fields read as recessed.
- **`matte-sheen`** and **`matte-sheen-solid`:** the user message bubble and badges. An
  almost invisible top-to-bottom sheen, fainter on solid fills.
- **`status-lamp`** and **`matte-orb`** with `orb-danger`, `orb-warning` or `orb-success`:
  small status dots and the recording button. A near-flat fill barely lighter at the top and
  an even hairline, with no glow.

The elevation tokens are `elevation-surface`, `elevation-control`, `elevation-raised`,
`elevation-inset` and `elevation-tooltip`. The drop shadows are shadow-plugin's smooth
stacks, five or so soft layers instead of one hard one: `elevation-surface` is its `md`,
`elevation-raised` its `lg`, and `elevation-control` only the near layers of its `sm`, so a
control sits close to the page. Two softer steps sit below them: `elevation-soft` for the chat form,
which floats only a little, and `elevation-faint` for the user message bubble, which barely
lifts. Dark mode deepens the same stacks. The plugin is imported
unprefixed, so Tailwind's own `shadow-xs` to `shadow-2xl` draw the smooth stacks too. Every
button carries `elevation-control` (`--button-shadow`); a ghost button only while hovered. Overlays behind dialogs are `surface-overlay`
with a 2px blur.

## Shapes

Corners are soft and consistent, set through roles: `control` (8px) for buttons, fields and
badges, `panel` (10px) for menus and dialogs, `card` (14px) for cards, `container` (16px) for
the sidebar, and `surface` (24px) for the chat form and message bubbles, rising to
`surface-lg` (32px) for the chat form on small screens. Status lamps, avatars and the round
icon buttons in the chat form are fully round.

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
  `toggle-background-on`.
- **Button groups:** joined buttons share one shadow, drawn by the group, and one border at
  each seam. A group separator replaces the seam border. Filled buttons draw a seam in their
  own text colour (`button-primary-seam`, `button-danger-seam`) instead.
- **Selection on a track:** a choice that moves along a track is a recessed track with the
  selection raised on it, never a row of outlined buttons. The track is `surface-muted` with
  the inset shade (`elevation-inset`); the selection is a `surface-lifted` chip with
  `elevation-control`, and unselected options are softened text with no fill. Tabs use the
  same track for their list, with the active tab raised (`tabs-*`); a `line` variant
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
  `edge-faint` hairline, so it separates from the page without a visible border.
- **Sidebar:** `sidebar-surface`, the muted fill at 60% with no inner highlight and the
  ambient edge. On small screens it keeps a plain `shadow-md` instead.
- **Fields:** `matte-well` with a 1px `line-field` outline and the standard 3px focus ring.
- **Floating panels:** every menu, popover, select list, hover card and drawer uses
  `matte-float`; dialogs use `matte-float-modal`.
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
- Do read component tokens in components, semantic tokens in shared styles, and primitives
  only in semantic tokens.
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

## Proposing changes

Visual changes are decided in Storybook before they reach the components.

1. **Prototype first.** Add a temporary story at `tests/stories/design-system/proposals/`
   titled `Design System/Proposals/<topic>`, with one story per question.
2. **Show current and proposed side by side.** Use `StateGallery` with two columns: each row
   is a pair of `StateGalleryItem`s labelled `Current` and `Proposed`, with a short note
   saying what differs. Render the current state with the real component, and the proposal
   in context next to the variants it sits among, not on its own.
3. **Leave the components alone.** Prototype with classes or a small global style in the
   story; the real components, tokens and app code change only once a proposal is chosen.
4. **Check both themes.** Review every proposal in light and dark mode before asking for a
   decision.
5. **Apply, then delete the proposal.** Once chosen, move the change into tokens and
   components, update the regular stories and this file, and remove the proposal story so
   it never ships.
