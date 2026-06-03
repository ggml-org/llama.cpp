import { createPreTransform } from './pre-transform';

/**
 * Converts svg code blocks to <pre class="svg-diagram"> for client-side rendering.
 */
export const rehypeSvgPre = createPreTransform('svg', 'svg-diagram');
