import { createPreTransform } from './pre-transform';
import { SVG_BLOCK_CLASS, SVG_LANGUAGE, XML_LANGUAGE, SVG_TAG_PREFIX } from '$lib/constants';

/**
 * Converts svg code blocks to <pre class="svg-block"> for client-side rendering.
 * Also claims xml blocks whose content starts with <svg, since models often emit
 * svg inside an xml fence.
 */
export const rehypeSvgPre = createPreTransform(
	[SVG_LANGUAGE, XML_LANGUAGE],
	SVG_BLOCK_CLASS,
	(text) => text.startsWith(SVG_TAG_PREFIX)
);
