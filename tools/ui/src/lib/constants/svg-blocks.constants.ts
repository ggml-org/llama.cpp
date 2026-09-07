/**
 * Constants for rendering svg code blocks inline.
 */
export const SVG = {
	// CSS classes applied to the inline svg block and its chrome.
	BLOCK_CLASS: 'svg-block',
	/**
	 * Shadow root style for the zoom dialog svg. Lets the svg grow past its
	 * intrinsic size so pan and zoom have room to work.
	 */
	DIALOG_SHADOW_STYLE:
		':host{display:inline-block}svg{min-height:min(50vh,12rem);min-width:min(80vw,20rem);max-width:none;max-height:none;height:auto;width:auto;display:block}',
	ID_ATTR: 'data-svg-id',

	/**
	 * Shadow root style for an inline svg block. Mirrors the centered, padded
	 * sizing the light dom used before the svg moved behind a shadow boundary.
	 */
	INLINE_SHADOW_STYLE:
		':host{display:block;width:100%;text-align:center}svg{display:block;margin:0 auto;width:auto;height:auto;max-width:100%;max-height:70vh;min-height:8rem;padding:3rem 1rem}',
	// Languages that mark a code block as svg content.
	LANGUAGE: 'svg',
	/**
	 * Hard size ceiling for a single inline svg block.
	 * Above this the source is left as raw text instead of being rendered.
	 */
	MAX_BYTES: 256 * 1024,

	RENDERED_ATTR: 'data-svg-rendered',
	/**
	 * DOMPurify config for untrusted svg coming from model output.
	 *
	 * foreignObject and script stay forbidden unconditionally, they are the only
	 * inline svg vectors that execute arbitrary html or js. Everything else is
	 * allowed for maximum rendering compatibility: href and xlink:href stay so
	 * use, image, a and animateMotion work, and DOMPurify still neutralizes
	 * javascript: and data: uri schemes natively. External resource refs are
	 * allowed by design on a local first tool, the user browser fetches them.
	 *
	 * The sanitized svg is always mounted inside a shadow root (see svg-shadow),
	 * so an author <style> stays scoped to that root and can not reach the page.
	 */
	SANITIZE_CONFIG: {
		FORBID_TAGS: ['foreignObject', 'script'],
		USE_PROFILES: { svg: true, svgFilters: true }
	},
	SCROLL_CONTAINER_CLASS: 'svg-scroll-container',

	// data-attributes used to stash per-block svg state on the DOM node.
	SOURCE_ATTR: 'data-svg-source',

	TAG_PREFIX: '<svg',

	WRAPPER_CLASS: 'svg-block-wrapper',

	XML_LANGUAGE: 'xml'
};
