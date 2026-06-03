import { createPreTransform } from './pre-transform';

/**
 * Converts svg code blocks to <pre class="svg-diagram"> for client-side rendering.
 * Also claims xml blocks whose content starts with <svg, since models often emit
 * svg inside an xml fence.
 */
export const rehypeSvgPre = createPreTransform(['svg', 'xml'], 'svg-diagram', (text) =>
	text.startsWith('<svg')
);
