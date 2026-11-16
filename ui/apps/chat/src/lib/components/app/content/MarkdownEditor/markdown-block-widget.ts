// A rendered block of the live preview: the HTML that MarkdownContent renders for its
// source (same pipeline, same code blocks and diagrams), its clicks, and selecting and
// copying its text.

import { EditorView, ViewPlugin, WidgetType } from '@codemirror/view';
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

export interface MarkdownBlockWidgetOptions {
	onPreviewCode?: (code: string, language: string) => void;
	onPreviewDiagram?: (svgHtml: string) => void;
	/**
	 * Anything outside the source that the rendered HTML depends on (e.g. how mention
	 * badges label paths): blocks with a different key render again.
	 */
	renderKey?: string;
}

interface Rendered {
	html: string;
	/** The block's height when it was last on screen. */
	height?: number;
}

const BLOCK_CLASS = 'cm-md-block';
const DRAG_THRESHOLD_PX = 4;
const RENDER_CACHE_LIMIT = 500;
// Height guesses for rendered blocks CodeMirror hasn't measured yet (off screen in a
// long document), which keep its scrollbar and scroll position steady
const ESTIMATED_ROW_PX = 24;
const ESTIMATED_ROW_CHARS = 80;
/** What each source rendered to, by render key and source; shared by every editor. */
const rendered = new Map<string, Rendered>();
/** The key of the render a block's DOM shows. */
const shownKey = new WeakMap<HTMLElement, string>();
/** The widget a block's DOM belongs to now; the DOM outlives the widget it was made for. */
const widgetOfDom = new WeakMap<HTMLElement, MarkdownBlockWidget>();

async function renderMarkdownBlock(source: string, key: string): Promise<string> {
	const processor = getMarkdownProcessor({});
	const markdown = preprocessLaTeX(splitGluedClosingCodeFences(source));
	const html = processor.stringify(await processor.run(processor.parse(markdown)));

	// bounded: the oldest render goes when the cache is full
	if (rendered.size >= RENDER_CACHE_LIMIT) rendered.delete(rendered.keys().next().value as string);

	rendered.set(key, { html });

	return html;
}

/** Rows `source` wraps to at a typical width: a cheap stand-in for its height. */
function estimatedRows(source: string): number {
	let rows = 0;

	for (const line of source.split('\n'))
		rows += Math.max(1, Math.ceil(line.length / ESTIMATED_ROW_CHARS));

	return rows;
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
function renderedSelection(view: EditorView): Range | null {
	const selection = document.getSelection();
	const anchor = selection?.anchorNode;

	if (!selection || selection.isCollapsed || !anchor || selection.rangeCount === 0) return null;

	const element = anchor instanceof Element ? anchor : anchor.parentElement;
	const block = element?.closest(`.${BLOCK_CLASS}`);

	return block && view.dom.contains(block) ? selection.getRangeAt(0) : null;
}

export class MarkdownBlockWidget extends WidgetType {
	// For blocks off screen: the height this block had when last on screen, or a guess
	// from its text before it ever was
	get estimatedHeight(): number {
		return rendered.get(this.key)?.height ?? estimatedRows(this.source) * ESTIMATED_ROW_PX;
	}

	private get key(): string {
		return `${this.options.renderKey ?? ''}\u0000${this.source}`;
	}

	constructor(
		readonly source: string,
		readonly gapRem: number,
		readonly options: MarkdownBlockWidgetOptions
	) {
		super();
	}

	// An equal widget keeps its DOM as it is, so only the blocks an edit changes redraw
	eq(other: MarkdownBlockWidget) {
		return (
			other.source === this.source &&
			other.gapRem === this.gapRem &&
			other.options.renderKey === this.options.renderKey
		);
	}

	// The editor leaves the rendered block alone (its own clicks, the reader's text
	// selection in it), see onMouseDown and copyRenderedSelection
	ignoreEvent() {
		return true;
	}

	toDOM(view: EditorView) {
		const dom = document.createElement('div');

		dom.className = `markdown-block ${BLOCK_CLASS}`;
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
		const { key } = this;

		widgetOfDom.set(dom, this);
		dom.style.paddingTop = `${this.gapRem}rem`;

		if (shownKey.get(dom) === key) return;

		const cached = rendered.get(key);

		if (cached) {
			this.show(dom, view, cached.html, key);

			return;
		}

		if (!dom.firstChild) {
			const fallback = document.createElement('p');

			fallback.textContent = this.source;
			dom.replaceChildren(fallback);
		}

		void renderMarkdownBlock(this.source, key).then((html) => {
			// a newer source may have reached this DOM while rendering
			if (widgetOfDom.get(dom)?.key === key) this.show(dom, view, html, key);
		});
	}

	// Measures the block in the editor's measure cycle (no forced layout) for
	// estimatedHeight; the measure also makes the editor take in the new height
	private measure(dom: HTMLElement, view: EditorView, key: string) {
		view.requestMeasure({
			key: dom,
			read: () => dom.offsetHeight,
			write: (height) => {
				const entry = rendered.get(key);

				if (entry && height > 0) entry.height = height;
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
	// the editor (ignoreEvent) and its copy keys (copyRenderedSelection) leave alone
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
		shownKey.set(dom, key);
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

// Copies the text the reader selected in rendered blocks, as text and as HTML. The
// editor's own copy would act on its selection instead (the cursor's whole line when
// it is empty), and the browser's stops at the first block, so this runs before both.
export const copyRenderedSelection = ViewPlugin.define((view) => {
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
