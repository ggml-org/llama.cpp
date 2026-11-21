export const MERMAID_WRAPPER_CLASS = 'mermaid-block-wrapper';
export const MERMAID_SCROLL_CONTAINER_CLASS = 'mermaid-scroll-container';
export const MERMAID_BLOCK_CLASS = 'mermaid';
export const MERMAID_ERROR_CLASS = 'mermaid-error';
// left plus right padding of a diagram's svg, see `pre.mermaid svg` in markdown-content.css
export const MERMAID_SVG_PADDING_X_REM = 2;

// class Pele puts on the root of every diagram it draws
export const PELE_SVG_CLASS = 'pele';
// the Pele colors app.css maps to the theme, written into a downloaded diagram
export const PELE_THEME_VARIABLES = [
	'--pele-bg',
	'--pele-fg',
	'--pele-muted',
	'--pele-line',
	'--pele-surface',
	'--pele-surface-alt',
	'--pele-border',
	'--pele-accent',
	'--pele-critical',
	'--pele-radius'
] as const;

export const MERMAID_LANGUAGE = 'mermaid';

export const MERMAID_SYNTAX_ATTR = 'data-mermaid-syntax';
export const MERMAID_ID_ATTR = 'data-mermaid-id';
export const MERMAID_RENDERED_ATTR = 'data-mermaid-rendered';
