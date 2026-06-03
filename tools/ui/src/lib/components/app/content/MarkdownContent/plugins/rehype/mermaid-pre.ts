import { createPreTransform } from './pre-transform';

/**
 * Converts mermaid code blocks to <pre class="mermaid"> for client-side rendering.
 */
export const rehypeMermaidPre = createPreTransform('mermaid', 'mermaid');
