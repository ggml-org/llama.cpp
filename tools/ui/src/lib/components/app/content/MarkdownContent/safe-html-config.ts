// Sanitization rules for raw HTML in model cards (MarkdownContent's
// allowHtml mode). Kept as its own module so the adversarial test in
// tests/client can import the exact shipped configuration.
import DOMPurify from 'dompurify';

// style is allowed for the KaTeX output, which lays its spans out with inline
// styles; a third party card can use the same attribute to draw a fixed
// overlay over the app, so positioned values are scrubbed
DOMPurify.addHook('afterSanitizeAttributes', (node) => {
	if (typeof (node as Element).getAttribute !== 'function') return;

	const style = (node as Element).getAttribute('style');

	if (!style || !/position\s*:\s*(fixed|absolute)/i.test(style)) return;

	(node as Element).setAttribute(
		'style',
		style.replace(/position\s*:\s*(fixed|absolute)\s*;?/gi, '')
	);
});

/**
 * Explicit DOMPurify allow list: the tags and attributes model cards may use
 * plus the KaTeX output (spans with inline styles and MathML). Everything
 * else - scripts, event handlers, forms, iframes, style tags - is stripped.
 */
export const SAFE_HTML_CONFIG = {
	ALLOWED_ATTR: [
		'class',
		'style',
		'id',
		'title',
		'role',
		'aria-hidden',
		'aria-label',
		'aria-describedby',
		'aria-expanded',
		'href',
		'target',
		'rel',
		'src',
		'srcset',
		'alt',
		'width',
		'height',
		'align',
		'valign',
		'colspan',
		'rowspan',
		'controls',
		'type',
		// MathML
		'xmlns',
		'display',
		'encoding',
		'mathvariant'
	],
	ALLOWED_TAGS: [
		// text and structure
		'a',
		'b',
		'i',
		'u',
		's',
		'em',
		'strong',
		'code',
		'pre',
		'kbd',
		'sub',
		'sup',
		'br',
		'hr',
		'p',
		'span',
		'div',
		'blockquote',
		'center',
		'ul',
		'ol',
		'li',
		'dl',
		'dt',
		'dd',
		'h1',
		'h2',
		'h3',
		'h4',
		'h5',
		'h6',
		'table',
		'thead',
		'tbody',
		'tr',
		'th',
		'td',
		'colgroup',
		'col',
		'caption',
		// media
		'img',
		'picture',
		'source',
		'figure',
		'figcaption',
		'video',
		'audio',
		// MathML, for the KaTeX output
		'math',
		'mrow',
		'mi',
		'mo',
		'mn',
		'ms',
		'mtext',
		'mfrac',
		'mroot',
		'msqrt',
		'msub',
		'msup',
		'msubsup',
		'munder',
		'mover',
		'mmultiscripts',
		'mtable',
		'mtr',
		'mtd',
		'mspace',
		'mglyph',
		'maligngroup',
		'malignmark',
		'mpadded',
		'mphantom',
		'mstyle',
		'semantics',
		'annotation'
	]
};
