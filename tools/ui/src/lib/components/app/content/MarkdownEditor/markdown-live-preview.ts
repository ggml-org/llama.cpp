// Live preview for MarkdownEditor: every top-level block the cursor is not in is
// replaced by the HTML that MarkdownContent renders for it (same pipeline, same
// stylesheet, same code blocks and diagrams), and the block being edited shows its raw
// text. The document text itself is never touched, so the prompt stays byte-exact.

import { syntaxTree, syntaxTreeAvailable } from '@codemirror/language';
import {
	type ChangeDesc,
	type EditorState,
	type Extension,
	type Range,
	StateEffect,
	StateField
} from '@codemirror/state';
import {
	Decoration,
	type DecorationSet,
	EditorView,
	ViewPlugin,
	WidgetType
} from '@codemirror/view';
import {
	getDiagramButtonAction,
	renderMermaidDiagrams,
	renderSvgDiagrams,
	toggleDiagramSource
} from '$lib/components/app/content/MarkdownContent/markdown-diagrams';
import { getMarkdownProcessor } from '$lib/components/app/content/MarkdownContent/markdown-processor';
import { getCodeInfoFromTarget } from '$lib/components/app/content/MarkdownContent/markdown-utils';
import { MERMAID_BLOCK_CLASS } from '$lib/constants';
import { ColorMode } from '$lib/enums';
import {
	copyCodeToClipboard,
	copyToClipboard,
	preprocessLaTeX,
	splitGluedClosingCodeFences
} from '$lib/utils';
import { mode } from 'mode-watcher';

export interface MarkdownLivePreviewOptions {
	/**
	 * Inline widgets for the raw text of the block being edited (e.g. file mention
	 * badges), given the state and the block's [from, to) range. They are atomic: the
	 * cursor steps over them and a delete removes them whole.
	 */
	editingWidgets?: (state: EditorState, from: number, to: number) => Range<Decoration>[];
	onPreviewCode?: (code: string, language: string) => void;
	onPreviewDiagram?: (svgHtml: string) => void;
	/**
	 * Anything outside the source that the rendered HTML depends on (e.g. how mention
	 * badges label paths). Blocks render again when it changes, see setRenderKey.
	 */
	renderKey?: string;
}

interface MarkdownBlock {
	from: number;
	/** End of the block's own text. */
	sourceTo: number;
	/** End of the block's range, including the blank lines that follow it. */
	to: number;
	name: string;
	gapRem: number;
	/** The block's own text, from..sourceTo. */
	source: string;
}

/** How a block shows: raw or rendered, and the gap above it in rem. */
interface BlockLook {
	gapRem: number;
	raw: boolean;
}

// [top, bottom] margins in rem of each block kind as a chat message renders it (each
// block in a .markdown-block of its own, so a heading is a first child). CodeMirror
// can not measure margins on block widgets, so the collapsed gap between two blocks is
// computed here and applied as padding. markdown-editor-margins.svelte.test.ts checks
// them against the stylesheet.
export const BLOCK_MARGINS_REM: Record<string, [number, number]> = {
	ATXHeading1: [0.5, 0.75],
	ATXHeading2: [0.5, 0.5],
	ATXHeading3: [0.5, 0.5],
	ATXHeading4: [0.5, 0.25],
	ATXHeading5: [0.5, 0.25],
	ATXHeading6: [0.5, 0.25],
	Blockquote: [1.5, 1.5],
	BulletList: [1, 1],
	CodeBlock: [1, 1],
	FencedCode: [1, 1],
	HorizontalRule: [2, 2],
	OrderedList: [1, 1],
	SetextHeading1: [0.5, 0.75],
	SetextHeading2: [0.5, 0.5],
	Table: [1, 1]
};
export const PARAGRAPH_MARGINS_REM: [number, number] = [1, 1];
const CODE_BLOCK_NAMES = new Set(['CodeBlock', 'FencedCode']);
// A link reference definition renders to nothing, so it stays raw to remain reachable
const ALWAYS_RAW_NAMES = new Set(['LinkReference']);
const MATH_FENCE = '$$';
const DRAG_THRESHOLD_PX = 4;
const RENDER_CACHE_LIMIT = 500;
// Height guesses for rendered blocks CodeMirror hasn't measured yet (off screen in a
// long document), which keep its scrollbar and scroll position steady
const ESTIMATED_ROW_PX = 24;
const ESTIMATED_ROW_CHARS = 80;
const renderCache = new Map<string, string>();
/** The measured height of a rendered block's content, by its cache key. */
const measuredHeight = new Map<string, number>();
const renderedKey = new WeakMap<HTMLElement, string>();
const widgetOfDom = new WeakMap<HTMLElement, MarkdownBlockWidget>();

function cacheKeyOf(source: string, renderKey: string): string {
	return `${renderKey}\u0000${source}`;
}

async function renderMarkdownBlock(source: string, renderKey: string): Promise<string> {
	const key = cacheKeyOf(source, renderKey);
	const cached = renderCache.get(key);

	if (cached !== undefined) return cached;

	const processor = getMarkdownProcessor({});
	const markdown = preprocessLaTeX(splitGluedClosingCodeFences(source));
	const html = processor.stringify(await processor.run(processor.parse(markdown)));

	remember(renderCache, key, html);

	return html;
}

/** Sets `key` in a bounded cache, dropping its oldest entry when full. */
function remember<T>(cache: Map<string, T>, key: string, value: T) {
	if (!cache.has(key) && cache.size >= RENDER_CACHE_LIMIT) {
		cache.delete(cache.keys().next().value as string);
	}

	cache.set(key, value);
}

function rangeKey(from: number, to: number): string {
	return `${from}:${to}`;
}

/** Rows `source` wraps to at a typical width: a cheap stand-in for its height. */
function estimatedRows(source: string): number {
	let rows = 0;

	for (const line of source.split('\n'))
		rows += Math.max(1, Math.ceil(line.length / ESTIMATED_ROW_CHARS));

	return rows;
}

function blockMargins(name: string): [number, number] {
	return BLOCK_MARGINS_REM[name] ?? PARAGRAPH_MARGINS_REM;
}

/** How many lines of `text` open or close a `$$` math block (`$$x$$` on one line does neither). */
function mathFenceCount(text: string): number {
	if (!text.includes(MATH_FENCE)) return 0;

	let count = 0;

	for (const line of text.split('\n')) {
		const trimmed = line.trim();

		if (!trimmed.startsWith(MATH_FENCE)) continue;

		if (trimmed.length > 2 * MATH_FENCE.length && trimmed.endsWith(MATH_FENCE)) continue;

		count++;
	}

	return count;
}

interface TopNode {
	/** Start of the node's first line. */
	from: number;
	name: string;
	/** End of the node's last line. */
	sourceTo: number;
	text?: string;
}

/**
 * A `$$` math block may hold blank lines, which split it into several top-level nodes
 * for the markdown parser; join them back into one block, as the renderer sees it.
 */
function joinMathBlocks(nodes: TopNode[], textOf: (node: TopNode) => string): TopNode[] {
	const opens = (node: TopNode) =>
		!CODE_BLOCK_NAMES.has(node.name) && mathFenceCount(textOf(node)) % 2 === 1;
	const joined: TopNode[] = [];

	for (let index = 0; index < nodes.length; index++) {
		const node = nodes[index];

		if (!opens(node)) {
			joined.push(node);

			continue;
		}

		let close = index + 1;

		while (close < nodes.length && !opens(nodes[close])) close++;

		// never closed: the block stays as the parser cut it
		if (close === nodes.length) {
			joined.push(node);

			continue;
		}

		joined.push({ from: node.from, name: node.name, sourceTo: nodes[close].sourceTo });
		index = close;
	}

	return joined;
}

/**
 * The texts of `blocks` that `changes` left alone, by their range after the changes,
 * so collecting the blocks again slices only what an edit touched.
 */
function untouchedTexts(blocks: MarkdownBlock[], changes: ChangeDesc | null): Map<string, string> {
	const texts = new Map<string, string>();

	for (const block of blocks) {
		if (!changes) {
			texts.set(rangeKey(block.from, block.sourceTo), block.source);
		} else if (!changes.touchesRange(block.from, block.sourceTo)) {
			texts.set(rangeKey(changes.mapPos(block.from), changes.mapPos(block.sourceTo)), block.source);
		}
	}

	return texts;
}

function collectBlocks(state: EditorState, texts?: Map<string, string>): MarkdownBlock[] {
	const { doc } = state;
	const tree = syntaxTree(state);
	// A long document may not be parsed to its end yet: its last parsed node may be cut
	// short, so it and the rest stay raw until the parser catches up (the tree change
	// rebuilds the preview)
	const complete = syntaxTreeAvailable(state, doc.length);
	const textOf = (node: TopNode) =>
		(node.text ??=
			texts?.get(rangeKey(node.from, node.sourceTo)) ?? doc.sliceString(node.from, node.sourceTo));

	let nodes: TopNode[] = [];

	for (let node = tree.topNode.firstChild; node; node = node.nextSibling) {
		nodes.push({
			from: doc.lineAt(node.from).from,
			name: node.name,
			sourceTo: doc.lineAt(node.to).to
		});
	}

	if (!complete) nodes.pop();

	nodes = joinMathBlocks(nodes, textOf);

	return nodes.map((node, index) => {
		const next = nodes[index + 1];
		const prev = nodes[index - 1];
		const to = next ? next.from - 1 : complete ? doc.length : node.sourceTo;
		const gapRem = prev ? Math.max(blockMargins(prev.name)[1], blockMargins(node.name)[0]) : 0;

		return {
			from: node.from,
			gapRem,
			name: node.name,
			source: textOf(node),
			sourceTo: node.sourceTo,
			to
		};
	});
}

let rootFontSize: number | undefined;

function rootFontSizePx(): number {
	rootFontSize ??= parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;

	return rootFontSize;
}

function caretAt(x: number, y: number): { node: Node; offset: number } | null {
	if (typeof document.caretPositionFromPoint === 'function') {
		const position = document.caretPositionFromPoint(x, y);

		return position && { node: position.offsetNode, offset: position.offset };
	}

	const range = document.caretRangeFromPoint?.(x, y);

	return range ? { node: range.startContainer, offset: range.startOffset } : null;
}

/** The page's selection when it is a non-empty one that starts in a rendered block of `view`. */
function renderedSelection(view: EditorView): ReturnType<Selection['getRangeAt']> | null {
	const selection = document.getSelection();
	const anchor = selection?.anchorNode;

	if (!selection || selection.isCollapsed || !anchor || selection.rangeCount === 0) return null;

	const element = anchor instanceof Element ? anchor : anchor.parentElement;
	const block = element?.closest('.cm-md-block');

	return block && view.dom.contains(block) ? selection.getRangeAt(0) : null;
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
	// For blocks off screen: the height this block had when last on screen, or a guess
	// from its text before it ever was
	get estimatedHeight(): number {
		const content =
			measuredHeight.get(cacheKeyOf(this.source, this.renderKey)) ??
			estimatedRows(this.source) * ESTIMATED_ROW_PX;

		return content + this.gapRem * rootFontSizePx();
	}

	constructor(
		readonly source: string,
		readonly gapRem: number,
		readonly renderKey: string,
		readonly options: MarkdownLivePreviewOptions
	) {
		super();
	}

	eq(other: MarkdownBlockWidget) {
		return (
			other.source === this.source &&
			other.gapRem === this.gapRem &&
			other.renderKey === this.renderKey
		);
	}

	// The editor leaves the rendered block alone (its own clicks, the reader's text
	// selection in it), see onMouseDown and copyRenderedSelection
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
		const key = cacheKeyOf(this.source, this.renderKey);

		widgetOfDom.set(dom, this);
		dom.style.paddingTop = `${this.gapRem}rem`;

		if (renderedKey.get(dom) === key) return;

		const cached = renderCache.get(key);

		if (cached !== undefined) {
			this.show(dom, view, cached, key);

			return;
		}

		if (!dom.firstChild) {
			const fallback = document.createElement('p');

			fallback.textContent = this.source;
			dom.replaceChildren(fallback);
		}

		void renderMarkdownBlock(this.source, this.renderKey).then((html) => {
			// a newer source may have reached this DOM while rendering
			const current = widgetOfDom.get(dom);

			if (!current || cacheKeyOf(current.source, current.renderKey) !== key) return;

			this.show(dom, view, html, key);
		});
	}

	// Measures the rendered content in the editor's measure cycle (no forced layout) for
	// estimatedHeight; the measure also makes the editor take in the new height
	private measure(dom: HTMLElement, view: EditorView, key: string) {
		view.requestMeasure({
			key: dom,
			read: () =>
				dom.isConnected
					? dom.offsetHeight - parseFloat(dom.style.paddingTop) * rootFontSizePx()
					: 0,
			write: (height) => {
				if (height > 0) remember(measuredHeight, key, height);
			}
		});
	}

	private onClick(event: MouseEvent) {
		const target = event.target as HTMLElement;

		if (toggleDiagramSource(target)) {
			event.preventDefault();
			event.stopPropagation();

			return;
		}

		const diagramAction = getDiagramButtonAction(target);

		if (diagramAction) {
			event.preventDefault();
			event.stopPropagation();

			if ('copy' in diagramAction) {
				void copyToClipboard(diagramAction.copy);
			} else {
				this.options.onPreviewDiagram?.(diagramAction.previewSvg);
			}

			return;
		}

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

	// A click on a rendered block opens it for editing; a drag selects its text, which
	// the editor (ignoreEvent) and its copy keys (see markdownLivePreview) leave alone
	private onMouseDown(event: MouseEvent, view: EditorView, dom: HTMLElement) {
		if (event.button !== 0) return;

		if ((event.target as HTMLElement).closest('a, button, input, summary')) return;

		// no native handling: it would focus the editor and turn a block raw mid-drag
		event.preventDefault();

		const anchor = caretAt(event.clientX, event.clientY);

		let selecting = false;

		const onMove = (move: MouseEvent) => {
			const distance = Math.hypot(move.clientX - event.clientX, move.clientY - event.clientY);

			if (!selecting && distance < DRAG_THRESHOLD_PX) return;

			selecting = true;

			const head = caretAt(move.clientX, move.clientY);

			if (anchor && head) {
				document
					.getSelection()
					?.setBaseAndExtent(anchor.node, anchor.offset, head.node, head.offset);
			}
		};
		const onUp = () => {
			window.removeEventListener('mousemove', onMove);
			window.removeEventListener('mouseup', onUp);

			if (!selecting) this.openForEditing(event, view, dom);
		};

		window.addEventListener('mousemove', onMove);
		window.addEventListener('mouseup', onUp);
	}

	// Shows the block's raw text with the cursor near the click
	private openForEditing(event: MouseEvent, view: EditorView, dom: HTMLElement) {
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

	private show(dom: HTMLElement, view: EditorView, html: string, key: string) {
		dom.innerHTML = html;
		renderedKey.set(dom, key);
		renderSvgDiagrams(dom);
		this.measure(dom, view, key);

		// mermaid draws asynchronously and changes the block's height when done
		if (dom.querySelector(`pre.${MERMAID_BLOCK_CLASS}`)) {
			void renderMermaidDiagrams(dom, mode.current === ColorMode.DARK).then(() =>
				this.measure(dom, view, key)
			);
		}
	}
}

interface LivePreview {
	atomic: DecorationSet;
	/** The top-level blocks the decorations were built from, reused while the text stays put. */
	blocks: MarkdownBlock[];
	decorations: DecorationSet;
	/** Which blocks show raw text, e.g. `"2,3"`; a cursor move within them changes nothing. */
	editingKey: string;
	focused: boolean;
	/** How each block shows in `decorations`. */
	looks: BlockLook[];
	renderKey: string;
}

function editingFlags(blocks: MarkdownBlock[], state: EditorState, focused: boolean): boolean[] {
	const { ranges } = state.selection;

	return blocks.map(
		(block) => focused && ranges.some((range) => range.from <= block.to && range.to >= block.from)
	);
}

function keyOf(flags: boolean[]): string {
	return flags.flatMap((editing, index) => (editing ? [index] : [])).join(',');
}

function looksOf(blocks: MarkdownBlock[], flags: boolean[]): BlockLook[] {
	const isRaw = (index: number) => flags[index] || ALWAYS_RAW_NAMES.has(blocks[index].name);

	return blocks.map((block, index) => {
		const prev = blocks[index - 1];
		// the blank lines after a raw block show as text already, so they are the gap
		const gapRem = prev && isRaw(index - 1) && prev.to > prev.sourceTo ? 0 : block.gapRem;

		return { gapRem, raw: isRaw(index) };
	});
}

/** The decorations that show one block: a rendered widget, or its raw text's spacing and line styles. */
function blockDecorations(
	state: EditorState,
	block: MarkdownBlock,
	look: BlockLook,
	renderKey: string,
	options: MarkdownLivePreviewOptions
): Range<Decoration>[] {
	if (!look.raw) {
		const widget = new MarkdownBlockWidget(block.source, look.gapRem, renderKey, options);

		return [Decoration.replace({ block: true, widget }).range(block.from, block.to)];
	}

	const { doc } = state;
	const decorations: Range<Decoration>[] = [];

	if (look.gapRem > 0) {
		decorations.push(
			Decoration.widget({ block: true, side: -1, widget: new SpacerWidget(look.gapRem) }).range(
				block.from
			)
		);
	}

	const lineClass = CODE_BLOCK_NAMES.has(block.name)
		? 'cm-md-raw-code'
		: ALWAYS_RAW_NAMES.has(block.name)
			? 'cm-md-raw-definition'
			: null;

	if (!lineClass) return decorations;

	const first = doc.lineAt(block.from).number;
	const last = doc.lineAt(block.sourceTo).number;

	for (let number = first; number <= last; number++) {
		const classes = [lineClass];

		if (number === first) classes.push(`${lineClass}-first`);

		if (number === last) classes.push(`${lineClass}-last`);

		decorations.push(Decoration.line({ class: classes.join(' ') }).range(doc.line(number).from));
	}

	return decorations;
}

/** Is `pos` in one of the sorted, disjoint [from, to] `ranges`? */
function inRanges(ranges: [number, number][], pos: number): boolean {
	let low = 0;
	let high = ranges.length - 1;

	while (low <= high) {
		const mid = (low + high) >> 1;

		if (pos < ranges[mid][0]) high = mid - 1;
		else if (pos > ranges[mid][1]) low = mid + 1;
		else return true;
	}

	return false;
}

/** The atomic inline widgets (e.g. mention badges) of a block showing raw text. */
function blockAtomic(
	state: EditorState,
	block: MarkdownBlock,
	look: BlockLook,
	options: MarkdownLivePreviewOptions
): Range<Decoration>[] {
	return look.raw && options.editingWidgets
		? options.editingWidgets(state, block.from, block.sourceTo)
		: [];
}

/** Moves `set` through `changes` and swaps its decorations in the `dirty` ranges for `added`. */
function patch(
	set: DecorationSet,
	changes: ChangeDesc | null,
	dirty: [number, number][],
	added: Range<Decoration>[]
): DecorationSet {
	const mapped = changes ? set.map(changes) : set;

	if (dirty.length === 0) return mapped;

	return mapped.update({
		add: added,
		filter: (from) => !inRanges(dirty, from),
		filterFrom: dirty[0][0],
		filterTo: dirty[dirty.length - 1][1],
		sort: true
	});
}

/**
 * The previous decorations, moved through `changes`, with new ones only for the blocks
 * that differ from before (their range, text, raw or rendered, or gap). Unchanged
 * blocks keep their decorations and widgets, so the editor neither compares nor
 * redraws them.
 */
function updateDecorations(
	previous: LivePreview,
	changes: ChangeDesc | null,
	state: EditorState,
	blocks: MarkdownBlock[],
	looks: BlockLook[],
	options: MarkdownLivePreviewOptions
): Pick<LivePreview, 'atomic' | 'decorations'> {
	// the previous blocks an edit left alone, by their range after it
	const kept = new Map<string, number>();

	previous.blocks.forEach((block, index) => {
		if (!changes) {
			kept.set(rangeKey(block.from, block.to), index);
		} else if (!changes.touchesRange(block.from, block.to)) {
			kept.set(rangeKey(changes.mapPos(block.from), changes.mapPos(block.to)), index);
		}
	});

	const dirty: [number, number][] = [];
	const added: Range<Decoration>[] = [];
	const addedAtomic: Range<Decoration>[] = [];
	const { length } = state.doc;

	// no block covers the start before the first one or a tail not parsed yet
	if (blocks.length === 0 || blocks[0].from > 0) {
		dirty.push([0, blocks.length === 0 ? length : blocks[0].from - 1]);
	}

	blocks.forEach((block, index) => {
		const was = kept.get(rangeKey(block.from, block.to));
		const same =
			was !== undefined &&
			previous.blocks[was].name === block.name &&
			previous.blocks[was].source === block.source &&
			previous.looks[was].raw === looks[index].raw &&
			previous.looks[was].gapRem === looks[index].gapRem;

		if (same) return;

		dirty.push([block.from, block.to]);
		added.push(...blockDecorations(state, block, looks[index], previous.renderKey, options));
		addedAtomic.push(...blockAtomic(state, block, looks[index], options));
	});

	const last = blocks.at(-1);

	if (last && last.to < length) dirty.push([last.to + 1, length]);

	return {
		atomic: patch(previous.atomic, changes, dirty, addedAtomic),
		decorations: patch(previous.decorations, changes, dirty, added)
	};
}

function buildPreview(
	state: EditorState,
	focused: boolean,
	renderKey: string,
	options: MarkdownLivePreviewOptions,
	blocks: MarkdownBlock[] = collectBlocks(state),
	previous?: { changes: ChangeDesc | null; preview: LivePreview }
): LivePreview {
	const flags = editingFlags(blocks, state, focused);
	const looks = looksOf(blocks, flags);
	const sets = previous
		? updateDecorations(previous.preview, previous.changes, state, blocks, looks, options)
		: {
				atomic: Decoration.set(
					blocks.flatMap((block, index) => blockAtomic(state, block, looks[index], options)),
					true
				),
				decorations: Decoration.set(
					blocks.flatMap((block, index) =>
						blockDecorations(state, block, looks[index], renderKey, options)
					),
					true
				)
			};

	return { ...sets, blocks, editingKey: keyOf(flags), focused, looks, renderKey };
}

const setFocused = StateEffect.define<boolean>();
const setRenderKeyEffect = StateEffect.define<string>();
// Outside the per-instance field, so a reconfigured preview still knows whether the
// editor has focus: keep markdownLivePreviewFocus out of the compartment holding it
const focusField = StateField.define<boolean>({
	create: () => false,
	update(focused, tr) {
		for (const effect of tr.effects) {
			if (effect.is(setFocused)) focused = effect.value;
		}

		return focused;
	}
});

export const markdownLivePreviewFocus: Extension = [
	focusField,
	EditorView.focusChangeEffect.of((_state, focusing) => setFocused.of(focusing))
];

// Copies the text the reader selected in rendered blocks, as text and as HTML. The
// editor's own copy would act on its selection instead (the cursor's whole line when
// it is empty), and the browser's stops at the first block, so this runs before both.
const copyRenderedSelection = ViewPlugin.define((view) => {
	const onCopy = (event: ClipboardEvent) => {
		const range = renderedSelection(view);

		if (!range || !event.clipboardData) return;

		const holder = document.createElement('div');

		holder.append(range.cloneContents());
		event.clipboardData.setData('text/html', holder.innerHTML);
		event.clipboardData.setData('text/plain', (holder.textContent ?? '').trim());
		event.preventDefault();
		event.stopPropagation();
	};

	view.dom.addEventListener('copy', onCopy, true);
	view.dom.addEventListener('cut', onCopy, true);

	return {
		destroy() {
			view.dom.removeEventListener('copy', onCopy, true);
			view.dom.removeEventListener('cut', onCopy, true);
		}
	};
});

/** Tells the preview whether the editor has focus; the editor does this on its own. */
export function setPreviewFocus(focused: boolean): StateEffect<boolean> {
	return setFocused.of(focused);
}

/** Renders the blocks again for a new render key, see MarkdownLivePreviewOptions. */
export function setRenderKey(renderKey: string): StateEffect<string> {
	return setRenderKeyEffect.of(renderKey);
}

export function markdownLivePreview(options: MarkdownLivePreviewOptions = {}): Extension {
	const field = StateField.define<LivePreview>({
		create: (state) =>
			buildPreview(
				state,
				state.field(focusField, false) ?? false,
				options.renderKey ?? '',
				options
			),
		provide: (f) => [
			EditorView.decorations.from(f, (value) => value.decorations),
			EditorView.decorations.from(f, (value) => value.atomic),
			EditorView.atomicRanges.from(f, (value) => () => value.atomic)
		],
		update(value, tr) {
			const focused = tr.state.field(focusField);

			let { renderKey } = value;

			for (const effect of tr.effects) {
				if (effect.is(setRenderKeyEffect)) renderKey = effect.value;
			}

			// every block renders differently
			if (renderKey !== value.renderKey) return buildPreview(tr.state, focused, renderKey, options);

			const treeChanged = syntaxTree(tr.state) !== syntaxTree(tr.startState);
			const changes = tr.docChanged ? tr.changes : null;

			if (!changes && !treeChanged) {
				if (!tr.selection && focused === value.focused) return value;

				// a cursor move or a focus change within the blocks showing raw text changes nothing
				const editingKey = keyOf(editingFlags(value.blocks, tr.state, focused));

				if (editingKey === value.editingKey)
					return focused === value.focused ? value : { ...value, focused };
			}

			const blocks =
				changes || treeChanged
					? collectBlocks(tr.state, untouchedTexts(value.blocks, changes))
					: value.blocks;

			return buildPreview(tr.state, focused, renderKey, options, blocks, {
				changes,
				preview: value
			});
		}
	});

	return [markdownLivePreviewFocus, copyRenderedSelection, field];
}
