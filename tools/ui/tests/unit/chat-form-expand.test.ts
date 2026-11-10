import {
	SHEET_AT_REST,
	sheetFrameAt,
	type SheetRects,
	shiftSheetRects,
	shouldExpandChatForm
} from '$lib/components/app/chat/ChatForm/chat-form-expand';
import { describe, expect, it } from 'vitest';

describe('shouldExpandChatForm', () => {
	it('ignores markdown markers on the line still being typed', () => {
		for (const text of ['*', '*really*', '-', '-5 degrees', '#', '#hashtag', '> ', '1.', '2024.']) {
			expect(shouldExpandChatForm(text), text).toBe(false);
		}

		expect(shouldExpandChatForm('# Title')).toBe(false);
		expect(shouldExpandChatForm('plain text\n- item')).toBe(false);
	});

	it('expands on a markdown block in a finished line', () => {
		expect(shouldExpandChatForm('# Title\n')).toBe(true);
		expect(shouldExpandChatForm('- item\nmore')).toBe(true);
		expect(shouldExpandChatForm('```js\nconst a = 1;')).toBe(true);
	});

	it('expands on several lines of plain text', () => {
		expect(shouldExpandChatForm('one\ntwo\nthree')).toBe(false);
		expect(shouldExpandChatForm('one\ntwo\nthree\nfour')).toBe(true);
	});
});

describe('sheetFrameAt', () => {
	// a phone screen, and the card floating above its bottom edge
	const sheet: SheetRects = {
		actionsTop: 780,
		card: { height: 844, left: 0, right: 390, top: 0 },
		padTop: 47
	};
	const card: SheetRects = {
		actionsTop: 770,
		card: { height: 104, left: 8, right: 382, top: 728 },
		padTop: 0
	};

	it('draws the sheet as the collapsed card', () => {
		const frame = sheetFrameAt(sheet, card);

		// raised by the status bar's padding, which is clipped off
		expect(frame.card.top).toBe('681px');
		expect(frame.card.clipPath).toBe('inset(47px 8px 693px 8px round 2rem)');
		// the actions land on the collapsed card's, whatever the card's own offset
		expect(frame.actions.top).toBe('-691px');
	});

	it('leaves the sheet alone when it already is the card', () => {
		const frame = sheetFrameAt(card, card);

		expect(frame.card.top).toBe(SHEET_AT_REST.card.top);
		expect(frame.actions.top).toBe(SHEET_AT_REST.actions.top);
		expect(frame.card.clipPath).toBe('inset(0px 0px 0px 0px round 2rem)');
	});

	it('moves the rects with the viewport', () => {
		const moved = shiftSheetRects(card, -30);

		expect(moved.actionsTop).toBe(740);
		expect(moved.card).toEqual({ ...card.card, top: 698 });
		expect(moved.padTop).toBe(card.padTop);
	});
});
