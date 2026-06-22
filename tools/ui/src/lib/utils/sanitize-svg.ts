import DOMPurify from 'dompurify';
import { SVG_MAX_BYTES, SVG_SANITIZE_CONFIG, SVG_TAG_PREFIX } from '$lib/constants';

/**
 * Sanitizes a raw svg string for safe inline rendering.
 * Returns the cleaned svg markup, or an empty string when the input is not a
 * usable svg, exceeds the size ceiling, or sanitizes to nothing. An empty
 * return tells the caller to keep the raw code block instead of rendering.
 */
export function sanitizeSvg(source: string): string {
	const trimmed = source.trim();

	if (!trimmed || trimmed.length > SVG_MAX_BYTES) return '';

	if (!trimmed.startsWith(SVG_TAG_PREFIX)) return '';

	const clean = DOMPurify.sanitize(trimmed, SVG_SANITIZE_CONFIG) as unknown as string;

	if (!clean || !clean.includes(SVG_TAG_PREFIX)) return '';

	return clean;
}
