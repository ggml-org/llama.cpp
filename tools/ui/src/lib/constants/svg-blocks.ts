export const SVG_WRAPPER_CLASS = 'svg-block-wrapper';
export const SVG_SCROLL_CONTAINER_CLASS = 'svg-scroll-container';
export const SVG_BLOCK_CLASS = 'svg-block';

export const SVG_LANGUAGE = 'svg';
export const XML_LANGUAGE = 'xml';
export const SVG_TAG_PREFIX = '<svg';

export const SVG_SOURCE_ATTR = 'data-svg-source';
export const SVG_ID_ATTR = 'data-svg-id';
export const SVG_RENDERED_ATTR = 'data-svg-rendered';

/**
 * Hard size ceiling for a single inline svg block.
 * Above this the source is left as raw text instead of being rendered.
 */
export const SVG_MAX_BYTES = 256 * 1024;

/**
 * Strict DOMPurify config for untrusted svg coming from model output.
 *
 * Forbidden tags close the classic svg xss vectors:
 * - foreignObject embeds arbitrary html
 * - script and a enable code execution and navigation
 * - style allows css @import and url() exfiltration
 * - use and image pull external resources (ssrf, tracking, referer leak)
 *
 * href and xlink:href are dropped entirely so no external reference survives.
 * Gradients and filters still work since they reference via fill="url(#id)".
 */
export const SVG_SANITIZE_CONFIG = {
	USE_PROFILES: { svg: true, svgFilters: true },
	FORBID_TAGS: ['foreignObject', 'script', 'style', 'a', 'use', 'image'],
	FORBID_ATTR: ['href', 'xlink:href']
};
