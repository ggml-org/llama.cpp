import { hasMarkdownStructure } from '$lib/components/app/content/MarkdownEditor/markdown-structure';
import { CHAT_FORM_EXPAND_MIN_LINES } from '$lib/constants';

/** The collapsed card's `rounded-4xl` on a phone. */
const COLLAPSED_CARD_RADIUS = '2rem';

export type SheetBox = { height: number; left: number; right: number; top: number };
/** Where the card and its actions are on screen, and the card's top padding. */
export type SheetRects = { actionsTop: number; card: SheetBox; padTop: number };
export type SheetFrame = { actions: Keyframe; card: Keyframe };

/**
 * Should a message expand the chat form on its own? Once it has several lines, or a
 * markdown block in its finished lines: the line still being typed does not count, as
 * a lone `-`, `#` or `1.` already parses as a block.
 */
export function shouldExpandChatForm(text: string): boolean {
	const finishedLines = text.slice(0, text.lastIndexOf('\n') + 1);

	return (
		text.split('\n').length >= CHAT_FORM_EXPAND_MIN_LINES || hasMarkdownStructure(finishedLines)
	);
}

/** The laid-out phone sheet as it is: full screen, square. */
export const SHEET_AT_REST: SheetFrame = {
	actions: { top: '0px' },
	card: { borderRadius: '0px', clipPath: 'inset(0 0 0 0 round 0px)', top: '0px' }
};

/**
 * The laid-out `sheet` drawn as the collapsed card at `card`: moved there, clipped to its
 * edges, its actions held at their place. Offsets, not transforms: WebKit composites
 * transforms apart from the clip, and the two drift.
 */
export function sheetFrameAt(sheet: SheetRects, card: SheetRects): SheetFrame {
	const rise = Math.max(0, sheet.padTop - card.padTop);
	const dy = card.card.top - sheet.card.top - rise;
	const clip = [
		rise,
		sheet.card.right - card.card.right,
		sheet.card.height - rise - card.card.height,
		card.card.left - sheet.card.left
	].map((px) => `${Math.max(0, px)}px`);

	return {
		actions: { top: `${card.actionsTop - sheet.actionsTop - dy}px` },
		card: {
			borderRadius: COLLAPSED_CARD_RADIUS,
			clipPath: `inset(${clip.join(' ')} round ${COLLAPSED_CARD_RADIUS})`,
			top: `${dy}px`
		}
	};
}

/** `rects` moved down the screen by `dy`. */
export function shiftSheetRects(rects: SheetRects, dy: number): SheetRects {
	return {
		...rects,
		actionsTop: rects.actionsTop + dy,
		card: { ...rects.card, top: rects.card.top + dy }
	};
}
