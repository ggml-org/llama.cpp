// Live preview for MarkdownEditor: every top-level block the cursor is not in is
// replaced by the HTML that MarkdownContent renders for it (same pipeline, same
// stylesheet, same code blocks and diagrams), and the block being edited shows its raw
// text. The document text itself is never touched, so the prompt stays byte-exact.

import {
	copyRenderedSelection,
	MarkdownBlockWidget,
	type MarkdownBlockWidgetOptions
} from './markdown-block-widget';
import { CODE_BLOCK_NAMES, collectBlocks, type MarkdownBlock } from './markdown-blocks';
import { syntaxTree } from '@codemirror/language';
import {
	type EditorState,
	type Extension,
	type Range,
	StateEffect,
	StateField
} from '@codemirror/state';
import { Decoration, type DecorationSet, EditorView, WidgetType } from '@codemirror/view';

export interface MarkdownLivePreviewOptions extends MarkdownBlockWidgetOptions {
	/**
	 * Inline widgets for the raw text of the block being edited (e.g. file mention
	 * badges), given the state and the block's [from, to) range. They are atomic: the
	 * cursor steps over them and a delete removes them whole.
	 */
	editingWidgets?: (state: EditorState, from: number, to: number) => Range<Decoration>[];
}

interface LivePreview {
	atomic: DecorationSet;
	/** The top-level blocks of the document, collected again only when its text changes. */
	blocks: MarkdownBlock[];
	decorations: DecorationSet;
	/** Which blocks show their raw text; a cursor move within them changes nothing. */
	raw: boolean[];
}

// Blocks that stay raw: a link reference definition renders to nothing, and a setext
// heading would hide its underline (`---` under a line of text is easy to type by accident)
const ALWAYS_RAW_NAMES = new Set(['LinkReference', 'SetextHeading1', 'SetextHeading2']);
const setFocused = StateEffect.define<boolean>();
// Whether the editor has focus. One field for every preview, so a preview that replaces
// another (new options) starts from the focus the editor has.
const focusField = StateField.define<boolean>({
	create: () => false,
	update(focused, tr) {
		for (const effect of tr.effects) {
			if (effect.is(setFocused)) focused = effect.value;
		}

		return focused;
	}
});
const focusTracking: Extension = [
	focusField,
	EditorView.focusChangeEffect.of((_state, focusing) => setFocused.of(focusing))
];

/** The gap above a raw block, which must stay outside the block's own background. */
class SpacerWidget extends WidgetType {
	constructor(readonly gapRem: number) {
		super();
	}

	eq(other: SpacerWidget) {
		return other.gapRem === this.gapRem;
	}

	toDOM() {
		const dom = document.createElement('div');

		dom.className = 'cm-md-spacer';
		dom.style.height = `${this.gapRem}rem`;

		return dom;
	}
}

/** The blocks that show their raw text: the ones the selection of a focused editor touches. */
function rawBlocks(state: EditorState, blocks: MarkdownBlock[]): boolean[] {
	const focused = state.field(focusField);
	const { ranges } = state.selection;

	return blocks.map(
		(block) =>
			ALWAYS_RAW_NAMES.has(block.name) ||
			(focused && ranges.some((range) => range.from <= block.to && range.to >= block.from))
	);
}

// Every block gets new decorations on every build. A widget equal to the one it
// replaces keeps its DOM (see MarkdownBlockWidget.eq), so the page only changes where
// the text did.
function buildPreview(
	state: EditorState,
	blocks: MarkdownBlock[],
	raw: boolean[],
	options: MarkdownLivePreviewOptions
): LivePreview {
	const decorations: Range<Decoration>[] = [];
	const atomic: Range<Decoration>[] = [];
	const { doc } = state;

	for (const [index, block] of blocks.entries()) {
		const prev = blocks[index - 1];
		// the blank lines after a raw block show as text already, so they are the gap
		const gapRem = prev && raw[index - 1] && prev.to > prev.sourceTo ? 0 : block.gapRem;

		if (!raw[index]) {
			const widget = new MarkdownBlockWidget(block.source, gapRem, options);

			decorations.push(Decoration.replace({ block: true, widget }).range(block.from, block.to));

			continue;
		}

		if (options.editingWidgets) {
			atomic.push(...options.editingWidgets(state, block.from, block.sourceTo));
		}

		if (gapRem > 0) {
			decorations.push(
				Decoration.widget({ block: true, side: -1, widget: new SpacerWidget(gapRem) }).range(
					block.from
				)
			);
		}

		const lineClass = CODE_BLOCK_NAMES.has(block.name)
			? 'cm-md-raw-code'
			: block.name === 'LinkReference'
				? 'cm-md-raw-definition'
				: null;

		if (!lineClass) continue;

		const first = doc.lineAt(block.from).number;
		const last = doc.lineAt(block.sourceTo).number;

		for (let number = first; number <= last; number++) {
			const classes = [lineClass];

			if (number === first) classes.push(`${lineClass}-first`);

			if (number === last) classes.push(`${lineClass}-last`);

			decorations.push(Decoration.line({ class: classes.join(' ') }).range(doc.line(number).from));
		}
	}

	return {
		atomic: Decoration.set(atomic, true),
		blocks,
		decorations: Decoration.set(decorations, true),
		raw
	};
}

export function markdownLivePreview(options: MarkdownLivePreviewOptions = {}): Extension {
	const field = StateField.define<LivePreview>({
		create(state) {
			const blocks = collectBlocks(state);

			return buildPreview(state, blocks, rawBlocks(state, blocks), options);
		},
		provide: (f) => [
			EditorView.decorations.from(f, (value) => value.decorations),
			EditorView.decorations.from(f, (value) => value.atomic),
			EditorView.atomicRanges.from(f, (value) => () => value.atomic)
		],
		update(value, tr) {
			const reparsed = tr.docChanged || syntaxTree(tr.state) !== syntaxTree(tr.startState);
			const blocks = reparsed ? collectBlocks(tr.state) : value.blocks;
			const raw = rawBlocks(tr.state, blocks);

			// a cursor move or a focus change that keeps the same blocks raw changes nothing
			if (!reparsed && raw.every((isRaw, index) => isRaw === value.raw[index])) return value;

			return buildPreview(tr.state, blocks, raw, options);
		}
	});

	return [focusTracking, copyRenderedSelection, field];
}

/**
 * The editor without the preview: every block shows its raw text, with the same inline
 * widgets (see MarkdownLivePreviewOptions.editingWidgets) over the whole document.
 */
export function markdownPlainText(
	editingWidgets?: MarkdownLivePreviewOptions['editingWidgets']
): Extension {
	if (!editingWidgets) return [];

	const widgetsOf = (state: EditorState) =>
		Decoration.set(editingWidgets(state, 0, state.doc.length), true);
	const field = StateField.define<DecorationSet>({
		create: widgetsOf,
		provide: (f) => [
			EditorView.decorations.from(f),
			EditorView.atomicRanges.from(f, (value) => () => value)
		],
		update: (value, tr) => (tr.docChanged ? widgetsOf(tr.state) : value)
	});

	return field;
}
