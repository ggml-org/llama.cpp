// The live preview updates its decorations block by block on each edit instead of
// building them all again. This drives random edits, cursor moves and focus changes
// and checks after each one that the result matches a full build from scratch, and that
// an edit leaves the other blocks' widgets as they were.

import { markdownLanguage } from '@codemirror/lang-markdown';
import { LanguageSupport } from '@codemirror/language';
import { EditorSelection, EditorState, type Range } from '@codemirror/state';
import { Decoration, type DecorationSet, EditorView, WidgetType } from '@codemirror/view';
import {
	markdownLivePreview,
	setPreviewFocus
} from '$lib/components/app/content/MarkdownEditor/markdown-live-preview';
import { describe, expect, it } from 'vitest';

class MarkWidget extends WidgetType {
	toDOM() {
		return document.createElement('span');
	}
}

// stands in for the mention badges: every `@x` in the blocks being edited
function editingWidgets(state: EditorState, from: number, to: number): Range<Decoration>[] {
	return [...state.doc.sliceString(from, to).matchAll(/@x/g)].map((match) =>
		Decoration.replace({ widget: new MarkWidget() }).range(
			from + match.index,
			from + match.index + 2
		)
	);
}

const extensions = [new LanguageSupport(markdownLanguage), markdownLivePreview({ editingWidgets })];
const PIECES = [
	'# Title\n\n',
	'para @x text\n\n',
	'- one\n- two\n\n',
	'```js\ncode\n```\n\n',
	'$$\na\n\nb\n$$\n\n',
	'> quote\n\n',
	'[r]: https://example.com\n\n',
	'| a |\n| - |\n| b |\n\n',
	'***\n\n',
	'text\n'
];
const INSERTS = ['a', '\n', '\n\n', '#', '- ', '```', '$$', '@x', ' ', '|', '>'];

// a small seeded generator, so a failure replays
function random(seed: number) {
	return (n: number) => {
		seed = (seed * 1103515245 + 12345) % 2 ** 31;

		return seed % n;
	};
}

function snapshot(state: EditorState): string[] {
	const lines: string[] = [];

	for (const source of state.facet(EditorView.decorations)) {
		if (typeof source === 'function') continue;

		for (let cursor = (source as DecorationSet).iter(); cursor.value; cursor.next()) {
			const { spec } = cursor.value;
			const widget = spec.widget as { source?: string; gapRem?: number } | undefined;

			lines.push(
				[
					cursor.from,
					cursor.to,
					spec.class ?? '',
					widget?.constructor.name ?? '',
					widget?.source ?? '',
					widget?.gapRem ?? ''
				].join('|')
			);
		}
	}

	return lines.sort();
}

function fresh(state: EditorState, focused: boolean): EditorState {
	return EditorState.create({ doc: state.doc, extensions, selection: state.selection }).update({
		effects: setPreviewFocus(focused)
	}).state;
}

describe('live preview incremental updates', () => {
	for (const seed of [1, 7, 42, 2026]) {
		it(`match a full build after every step (seed ${seed})`, () => {
			const pick = random(seed);

			let focused = false;
			let state = EditorState.create({
				doc: Array.from({ length: 8 }, () => PIECES[pick(PIECES.length)]).join(''),
				extensions
			});

			for (let step = 0; step < 150; step++) {
				const { length } = state.doc;
				const kind = pick(5);
				const at = pick(length + 1);

				if (kind === 0) {
					state = state.update({
						changes: { from: at, insert: INSERTS[pick(INSERTS.length)] }
					}).state;
				} else if (kind === 1 && length > 0) {
					state = state.update({
						changes: { from: at, to: Math.min(length, at + 1 + pick(6)) }
					}).state;
				} else if (kind === 2) {
					state = state.update({
						changes: { from: at, insert: PIECES[pick(PIECES.length)] }
					}).state;
				} else if (kind === 3) {
					state = state.update({ selection: EditorSelection.cursor(at) }).state;
				} else {
					focused = !focused;
					state = state.update({ effects: setPreviewFocus(focused) }).state;
				}

				expect(snapshot(state), `step ${step}`).toEqual(snapshot(fresh(state, focused)));
			}
		});
	}

	it('keeps the widgets of the blocks an edit leaves alone', () => {
		const widgetAt = (state: EditorState, pos: number) => {
			for (const source of state.facet(EditorView.decorations)) {
				if (typeof source === 'function') continue;

				for (let cursor = (source as DecorationSet).iter(pos); cursor.value; cursor.next()) {
					if (cursor.from === pos) return cursor.value.spec.widget;
				}
			}
		};
		const before = EditorState.create({ doc: 'first\n\nsecond\n\nthird', extensions });
		const after = before.update({ changes: { from: 'first\n\nsec'.length, insert: 'X' } }).state;

		expect(widgetAt(after, 0)).toBe(widgetAt(before, 0));
		expect(widgetAt(after, 'first\n\nsecXond\n\n'.length)).toBe(
			widgetAt(before, 'first\n\nsecond\n\n'.length)
		);
		expect(widgetAt(after, 'first\n\n'.length)).not.toBe(widgetAt(before, 'first\n\n'.length));
	});
});
