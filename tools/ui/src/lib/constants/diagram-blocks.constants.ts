// Shared constants for diagram blocks (mermaid and svg) that toggle between a
// rendered view and a source view. The wrapper carries the active mode, css
// drives the visibility, the click handler only flips the attribute.

export const DIAGRAM_VIEW_MODE_ATTR = 'data-view-mode';
export const DIAGRAM_VIEW_RENDERED = 'rendered';
export const DIAGRAM_VIEW_SOURCE = 'source';
export const DIAGRAM_SOURCE_CLASS = 'diagram-source';
export const TOGGLE_SOURCE_BTN_CLASS = 'toggle-source-btn';
