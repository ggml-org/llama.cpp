export const LINE_BREAK = /\r?\n/;

export const PHRASE_PARENTS = new Set([
	'paragraph',
	'heading',
	'emphasis',
	'strong',
	'delete',
	'link',
	'linkReference',
	'tableCell'
]);

export const NBSP = '\u00a0';
export const TAB_AS_SPACES = NBSP.repeat(4);
