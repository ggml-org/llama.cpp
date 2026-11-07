// Live preview for the markdown editors (notebook, chat form): every top-level block
// the cursor is not in is replaced by the HTML that MarkdownContent renders for it
// (same pipeline, same stylesheet, same code blocks), and the block being edited shows
// its raw text.
// The document text itself is never touched, so the prompt stays byte-exact.

import { ensureSyntaxTree, syntaxTree } from '@codemirror/language';
import {
	type EditorState,
	type Extension,
	type Range,
	StateEffect,
	StateField
} from '@codemirror/state';
import { Decoration, type DecorationSet, EditorView, WidgetType } from '@codemirror/view';
import { getMarkdownProcessor } from '$lib/components/app/content/MarkdownContent/markdown-processor';
import { getCodeInfoFromTarget } from '$lib/components/app/content/MarkdownContent/markdown-utils';
import { copyCodeToClipboard, preprocessLaTeX, splitGluedClosingCodeFences } from '$lib/utils';

export interface MarkdownLivePreviewOptions {
	/**
	 * Inline widgets for the raw text of the block being edited (e.g. file mention
	 * badges), given its source and document offset. They are atomic: the cursor
	 * steps over them and a delete removes them whole.
	 */
	editingWidgets?: (source: string, from: number) => Range<Decoration>[];
	onPreviewCode?: (code: string, language: string) => void;
}

interface MarkdownBlock {
	from: number;
	/** End of the block's own text. */
	sourceTo: number;
	/** End of the block's range, including the blank lines that follow it. */
	to: number;
	name: string;
	gapRem: number;
}

// [top, bottom] margins in rem of each block kind in markdown-content.css. CodeMirror
// can not measure margins on block widgets, so the collapsed gap between two blocks is
// computed here and applied as padding.
const BLOCK_MARGINS_REM: Record<string, [number, number]> = {
	ATXHeading1: [1.5, 0.75],
	ATXHeading2: [1.25, 0.5],
	ATXHeading3: [1.5, 0.5],
	ATXHeading4: [0.75, 0.25],
	ATXHeading5: [0.5, 0.25],
	ATXHeading6: [0.5, 0.25],
	Blockquote: [1.5, 1.5],
	BulletList: [0, 1],
	CodeBlock: [1, 1],
	FencedCode: [1, 1],
	HorizontalRule: [2, 2],
	OrderedList: [0, 1],
	SetextHeading1: [1.5, 0.75],
	SetextHeading2: [1.25, 0.5],
	Table: [1.5, 1.5]
};
const PARAGRAPH_MARGINS_REM: [number, number] = [1, 1];
const CODE_BLOCK_NAMES = new Set(['CodeBlock', 'FencedCode']);
const RENDER_CACHE_LIMIT = 500;
const PARSE_TIMEOUT_MS = 50;
const renderCache = new Map<string, string>();
const renderedSource = new WeakMap<HTMLElement, string>();
const widgetOfDom = new WeakMap<HTMLElement, MarkdownBlockWidget>();

async function renderMarkdownBlock(source: string): Promise<string> {
	const cached = renderCache.get(source);

	if (cached !== undefined) return cached;

	const processor = getMarkdownProcessor({});
	const markdown = preprocessLaTeX(splitGluedClosingCodeFences(source));
	const html = processor.stringify(await processor.run(processor.parse(markdown)));

	if (renderCache.size >= RENDER_CACHE_LIMIT) {
		renderCache.delete(renderCache.keys().next().value as string);
	}

	renderCache.set(source, html);

	return html;
}

function blockMargins(name: string): [number, number] {
	return BLOCK_MARGINS_REM[name] ?? PARAGRAPH_MARGINS_REM;
}

function collectBlocks(state: EditorState): MarkdownBlock[] {
	const { doc } = state;
	const tree = ensureSyntaxTree(state, doc.length, PARSE_TIMEOUT_MS) ?? syntaxTree(state);
	const nodes: { from: number; name: string; to: number }[] = [];

	for (let node = tree.topNode.firstChild; node; node = node.nextSibling) {
		nodes.push({ from: node.from, name: node.name, to: node.to });
	}

	return nodes.map((node, index) => {
		const next = nodes[index + 1];
		const prev = nodes[index - 1];
		const from = doc.lineAt(node.from).from;
		const to = next ? doc.lineAt(next.from).from - 1 : doc.length;
		const gapRem = prev ? Math.max(blockMargins(prev.name)[1], blockMargins(node.name)[0]) : 0;

		return { from, gapRem, name: node.name, sourceTo: doc.lineAt(node.to).to, to };
	});
}

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

class MarkdownBlockWidget extends WidgetType {
	constructor(
		readonly source: string,
		readonly gapRem: number,
		readonly options: MarkdownLivePreviewOptions
	) {
		super();
	}

	eq(other: MarkdownBlockWidget) {
		return other.source === this.source && other.gapRem === this.gapRem;
	}

	ignoreEvent() {
		return true;
	}

	toDOM(view: EditorView) {
		const dom = document.createElement('div');

		dom.className = 'markdown-block cm-md-block';
		dom.addEventListener('mousedown', (event) =>
			widgetOfDom.get(dom)?.onMouseDown(event, view, dom)
		);
		dom.addEventListener('click', (event) => widgetOfDom.get(dom)?.onClick(event));
		this.fill(dom, view);

		return dom;
	}

	// Reuse the DOM when only the source changed (e.g. streamed tokens), keeping the
	// previous render on screen until the new one is ready
	updateDOM(dom: HTMLElement, view: EditorView) {
		this.fill(dom, view);

		return true;
	}

	private fill(dom: HTMLElement, view: EditorView) {
		widgetOfDom.set(dom, this);
		dom.style.paddingTop = `${this.gapRem}rem`;

		if (renderedSource.get(dom) === this.source) return;

		const cached = renderCache.get(this.source);

		if (cached !== undefined) {
			dom.innerHTML = cached;
			renderedSource.set(dom, this.source);

			return;
		}

		if (!dom.firstChild) {
			const fallback = document.createElement('p');

			fallback.textContent = this.source;
			dom.replaceChildren(fallback);
		}

		const source = this.source;

		void renderMarkdownBlock(source).then((html) => {
			// a newer source may have reached this DOM while rendering
			if (widgetOfDom.get(dom)?.source !== source) return;

			dom.innerHTML = html;
			renderedSource.set(dom, source);
			view.requestMeasure();
		});
	}

	private onClick(event: MouseEvent) {
		const target = event.target as HTMLElement;
		const copyButton = target.closest<HTMLElement>('.copy-code-btn');
		const previewButton = target.closest<HTMLElement>('.preview-code-btn');
		const button = copyButton ?? previewButton;

		if (!button) return;

		event.preventDefault();
		event.stopPropagation();

		const info = getCodeInfoFromTarget(button);

		if (!info) return;

		if (copyButton) {
			void copyCodeToClipboard(info.rawCode);
		} else {
			this.options.onPreviewCode?.(info.rawCode, info.language);
		}
	}

	// Clicking a rendered block opens it for editing, with the cursor near the click
	private onMouseDown(event: MouseEvent, view: EditorView, dom: HTMLElement) {
		if ((event.target as HTMLElement).closest('a, button, input, summary')) return;

		event.preventDefault();

		const from = view.posAtDOM(dom);
		const to = from + this.source.length;

		view.dispatch({ selection: { anchor: from } });
		view.focus();

		// The raw block has a different layout: map the click once the editor has
		// re-measured it (its height map settles over the next two frames)
		requestAnimationFrame(() =>
			requestAnimationFrame(() => {
				const pos = view.posAtCoords({ x: event.clientX, y: event.clientY });

				if (pos !== null && pos >= from && pos <= to) {
					view.dispatch({ selection: { anchor: pos } });
				}
			})
		);
	}
}

interface LivePreview {
	atomic: DecorationSet;
	decorations: DecorationSet;
	focused: boolean;
}

function buildDecorations(
	state: EditorState,
	focused: boolean,
	options: MarkdownLivePreviewOptions
): Omit<LivePreview, 'focused'> {
	const decorations: Range<Decoration>[] = [];
	const atomic: Range<Decoration>[] = [];
	const { doc, selection } = state;

	for (const block of collectBlocks(state)) {
		const isEditing =
			focused && selection.ranges.some((range) => range.from <= block.to && range.to >= block.from);

		if (!isEditing) {
			const source = doc.sliceString(block.from, block.sourceTo);
			const widget = new MarkdownBlockWidget(source, block.gapRem, options);

			decorations.push(Decoration.replace({ block: true, widget }).range(block.from, block.to));

			continue;
		}

		if (options.editingWidgets) {
			atomic.push(
				...options.editingWidgets(doc.sliceString(block.from, block.sourceTo), block.from)
			);
		}

		if (block.gapRem > 0) {
			decorations.push(
				Decoration.widget({ block: true, side: -1, widget: new SpacerWidget(block.gapRem) }).range(
					block.from
				)
			);
		}

		if (CODE_BLOCK_NAMES.has(block.name)) {
			const first = doc.lineAt(block.from).number;
			const last = doc.lineAt(block.sourceTo).number;

			for (let number = first; number <= last; number++) {
				const classes = ['cm-md-raw-code'];

				if (number === first) classes.push('cm-md-raw-code-first');

				if (number === last) classes.push('cm-md-raw-code-last');

				decorations.push(
					Decoration.line({ class: classes.join(' ') }).range(doc.line(number).from)
				);
			}
		}
	}

	return { atomic: Decoration.set(atomic, true), decorations: Decoration.set(decorations, true) };
}

const setFocused = StateEffect.define<boolean>();

export function markdownLivePreview(options: MarkdownLivePreviewOptions = {}): Extension {
	const field = StateField.define<LivePreview>({
		create: (state) => ({ ...buildDecorations(state, false, options), focused: false }),
		provide: (f) => [
			EditorView.decorations.from(f, (value) => value.decorations),
			EditorView.decorations.from(f, (value) => value.atomic),
			EditorView.atomicRanges.from(f, (value) => () => value.atomic)
		],
		update(value, tr) {
			let { focused } = value;

			for (const effect of tr.effects) {
				if (effect.is(setFocused)) focused = effect.value;
			}

			const treeChanged = syntaxTree(tr.state) !== syntaxTree(tr.startState);

			if (!tr.docChanged && !tr.selection && !treeChanged && focused === value.focused) {
				return value;
			}

			return { ...buildDecorations(tr.state, focused, options), focused };
		}
	});

	return [field, EditorView.focusChangeEffect.of((_state, focusing) => setFocused.of(focusing))];
}
