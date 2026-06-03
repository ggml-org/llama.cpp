import DOMPurify from 'dompurify';

/**
 * Hard size ceiling for a single inline svg block.
 * Above this the source is left as raw text instead of being rendered.
 */
const MAX_SVG_BYTES = 256 * 1024;

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
const SVG_CONFIG = {
	USE_PROFILES: { svg: true, svgFilters: true },
	FORBID_TAGS: ['foreignObject', 'script', 'style', 'a', 'use', 'image'],
	FORBID_ATTR: ['href', 'xlink:href']
};

/**
 * Sanitizes a raw svg string for safe inline rendering.
 * Returns the cleaned svg markup, or an empty string when the input is not a
 * usable svg, exceeds the size ceiling, or sanitizes to nothing. An empty
 * return tells the caller to keep the raw code block instead of rendering.
 */
export function sanitizeSvg(source: string): string {
	const trimmed = source.trim();

	if (!trimmed || trimmed.length > MAX_SVG_BYTES) return '';

	if (!trimmed.startsWith('<svg')) return '';

	const clean = DOMPurify.sanitize(trimmed, SVG_CONFIG) as unknown as string;

	if (!clean || !clean.includes('<svg')) return '';

	return clean;
}
