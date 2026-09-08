import { SVG } from '$lib/constants';
import DOMPurify from 'dompurify';

/**
 * Sanitizes a raw svg string for safe inline rendering.
 * Returns the cleaned svg markup, or an empty string when the input is not a
 * usable svg, exceeds the size ceiling, or sanitizes to nothing. An empty
 * return tells the caller to keep the raw code block instead of rendering.
 */
export function sanitizeSvg(source: string): string {
	const trimmed = source.trim();

	if (!trimmed || trimmed.length > SVG.MAX_BYTES) return '';

	if (!trimmed.startsWith(SVG.TAG_PREFIX)) return '';

	const clean = DOMPurify.sanitize(trimmed, SVG.SANITIZE_CONFIG) as unknown as string;

	if (!clean || !clean.includes(SVG.TAG_PREFIX)) return '';

	return clean;
}
