// Guards the MarkdownEditor contract beyond the chat form: which edits reach onChange,
// undo of a programmatic replace, props changed after mount, blocks the live preview
// keeps raw or joins, the gap after a raw block, and selecting text in rendered blocks.

import { undo } from '@codemirror/commands';
import { EditorView, type WidgetType } from '@codemirror/view';
import MarkdownEditor from '$lib/components/app/content/MarkdownEditor/MarkdownEditor.svelte';
import { SVG } from '$lib/constants';
import { tick } from 'svelte';
import { describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

function editorIn(container: HTMLElement): EditorView {
	const el = container.querySelector<HTMLElement>('.cm-content');
	const view = el && EditorView.findFromDOM(el);

	if (!view) throw new Error('editor not rendered');

	return view;
}

async function settle() {
	await tick();
	await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
}

function mouse(type: string, target: Element, x: number, y: number) {
	target.dispatchEvent(
		new MouseEvent(type, { bubbles: true, button: 0, cancelable: true, clientX: x, clientY: y })
	);
}

describe('MarkdownEditor', () => {
	it('reports every edit but the syncs from value', async () => {
		const onChange = vi.fn();
		const screen = render(MarkdownEditor, { onChange, value: 'a' });
		const editor = editorIn(screen.container);

		await screen.rerender({ value: 'ab' });
		await settle();

		expect(editor.state.doc.toString()).toBe('ab');
		expect(onChange).not.toHaveBeenCalled();

		// an edit with no user event, e.g. from another extension
		editor.dispatch({ changes: { from: 0, insert: 'x' } });

		expect(onChange).toHaveBeenCalledWith('xab');
	});

	it('undoes a replaced text in one step', async () => {
		const onChange = vi.fn();
		const screen = render(MarkdownEditor, { onChange, value: 'see @no' });
		const editor = editorIn(screen.container);

		screen.component.replaceText('see [notes.md](file:///notes.md) ', 33);

		expect(onChange).toHaveBeenLastCalledWith('see [notes.md](file:///notes.md) ');
		expect(editor.state.selection.main.head).toBe(33);

		undo(editor);

		expect(editor.state.doc.toString()).toBe('see @no');
	});

	it('follows placeholder and label changes after mount', async () => {
		const screen = render(MarkdownEditor, { placeholder: 'First', value: '' });
		const editor = editorIn(screen.container);

		await screen.rerender({ placeholder: 'Second' });
		await settle();

		expect(screen.container.querySelector('.cm-placeholder')?.textContent).toBe('Second');
		expect(editor.contentDOM.getAttribute('aria-label')).toBe('Second');
	});

	it('reconfigures only when a prop changes, not when it mounts', async () => {
		const reconfigured = vi.fn();
		const extensions = [
			EditorView.updateListener.of((update) => {
				if (update.transactions.some((tr) => tr.reconfigured)) reconfigured();
			})
		];
		const screen = render(MarkdownEditor, { extensions, placeholder: 'First', value: '' });

		await settle();

		expect(reconfigured).not.toHaveBeenCalled();

		await screen.rerender({ value: 'typed' });
		await settle();

		expect(reconfigured).not.toHaveBeenCalled();

		await screen.rerender({ placeholder: 'Second' });
		await settle();

		expect(reconfigured).toHaveBeenCalledTimes(1);
	});

	it('keeps the block being edited raw when the render key changes', async () => {
		const screen = render(MarkdownEditor, { renderKey: 'short', value: 'one\n\ntwo' });
		const editor = editorIn(screen.container);
		const blocks = () => [...screen.container.querySelectorAll<HTMLElement>('.cm-md-block')];

		await vi.waitFor(() => expect(blocks()).toHaveLength(2));

		editor.focus();
		editor.dispatch({ selection: { anchor: 0 } });

		await vi.waitFor(() => expect(blocks().map((block) => block.textContent)).toEqual(['two']));

		await screen.rerender({ renderKey: 'full' });
		await settle();

		expect(blocks().map((block) => block.textContent)).toEqual(['two']);
	});

	it('leaves the other rendered blocks untouched while a block is typed in', async () => {
		const screen = render(MarkdownEditor, { value: '# One\n\ntwo\n\n- three\n\nfour' });
		const editor = editorIn(screen.container);
		const blocks = () => [...screen.container.querySelectorAll<HTMLElement>('.cm-md-block')];

		await vi.waitFor(() =>
			expect(blocks().map((block) => block.textContent?.trim())).toEqual([
				'One',
				'two',
				'three',
				'four'
			])
		);

		editor.focus();
		editor.dispatch({ selection: { anchor: '# One\n\nt'.length } });

		await vi.waitFor(() => expect(blocks()).toHaveLength(3));

		const others = blocks();
		const mutations: MutationRecord[] = [];
		const observer = new MutationObserver((records) => mutations.push(...records));

		for (const block of others) {
			observer.observe(block, {
				attributes: true,
				characterData: true,
				childList: true,
				subtree: true
			});
		}

		for (const char of 'yped') {
			editor.dispatch(editor.state.replaceSelection(char), { userEvent: 'input.type' });
			await settle();
		}

		observer.disconnect();

		expect(editor.state.doc.toString()).toBe('# One\n\ntypedwo\n\n- three\n\nfour');
		expect(blocks()).toEqual(others);
		expect(mutations).toEqual([]);
	});

	it('keeps a link reference definition raw', async () => {
		const screen = render(MarkdownEditor, { value: 'see [docs][d]\n\n[d]: https://example.com' });

		await vi.waitFor(() => {
			expect(screen.container.querySelectorAll('.cm-md-block')).toHaveLength(1);
			expect(screen.container.querySelector('.cm-md-raw-definition')?.textContent).toBe(
				'[d]: https://example.com'
			);
		});
	});

	it('keeps a setext heading raw, so its underline never disappears', async () => {
		const screen = render(MarkdownEditor, { value: 'test\n---\nafter' });

		await vi.waitFor(() => {
			expect(
				[...screen.container.querySelectorAll('.cm-md-block')].map((b) => b.textContent)
			).toEqual(['after']);
			expect([...screen.container.querySelectorAll('.cm-line')].map((l) => l.textContent)).toEqual([
				'test',
				'---'
			]);
		});
	});

	it('renders a $$ math block with a blank line as one block', async () => {
		const screen = render(MarkdownEditor, { value: '$$\na\n\nb\n$$' });

		await vi.waitFor(() => {
			const blocks = screen.container.querySelectorAll('.cm-md-block');

			expect(blocks).toHaveLength(1);
			expect(blocks[0].querySelector('.katex')).not.toBeNull();
		});
	});

	it('draws an svg block like a chat message', async () => {
		const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect/></svg>';
		const screen = render(MarkdownEditor, { value: `\`\`\`svg\n${svg}\n\`\`\`` });

		await vi.waitFor(() => {
			expect(screen.container.querySelector(`.cm-md-block [${SVG.RENDERED_ATTR}]`)).not.toBeNull();
		});
	});

	it('lets the blank lines after a raw block be the gap to the next one', async () => {
		const screen = render(MarkdownEditor, { value: 'first\n\nsecond' });
		const editor = editorIn(screen.container);
		const second = () => screen.container.querySelector<HTMLElement>('.cm-md-block');

		await vi.waitFor(() =>
			expect(screen.container.querySelectorAll('.cm-md-block')).toHaveLength(2)
		);

		editor.focus();
		editor.dispatch({ selection: { anchor: 0 } });
		await settle();

		expect(screen.container.querySelectorAll('.cm-md-block')).toHaveLength(1);
		expect(second()?.style.paddingTop).toBe('0rem');
	});

	it('estimates a rendered block by its measured height', async () => {
		const screen = render(MarkdownEditor, { value: 'A paragraph\nover two lines' });
		const editor = editorIn(screen.container);
		const widget = () => {
			for (const source of editor.state.facet(EditorView.decorations)) {
				if (typeof source === 'function') continue;

				const cursor = source.iter();

				if (cursor.value?.spec.widget) return cursor.value.spec.widget as WidgetType;
			}
		};

		await vi.waitFor(() => {
			const block = screen.container.querySelector<HTMLElement>('.cm-md-block');

			expect(block?.querySelector('p')).not.toBeNull();
			expect(widget()?.estimatedHeight).toBe(block!.offsetHeight);
		});
	});

	it('copies a selection across rendered blocks', async () => {
		const screen = render(MarkdownEditor, { value: '# Plan\n\n- one\n- two' });
		const blocks = () => screen.container.querySelectorAll('.cm-md-block');

		await vi.waitFor(() => expect(blocks()[1]?.querySelectorAll('li')).toHaveLength(2));

		const range = document.createRange();

		range.setStart(blocks()[0].querySelector('h1')!.firstChild!, 0);
		range.setEnd(blocks()[1].querySelectorAll('li')[1].firstChild!, 3);
		document.getSelection()?.removeAllRanges();
		document.getSelection()?.addRange(range);

		const clipboardData = new DataTransfer();

		blocks()[0].dispatchEvent(
			new ClipboardEvent('copy', { bubbles: true, cancelable: true, clipboardData })
		);

		expect(clipboardData.getData('text/plain').replace(/\s+/g, ' ')).toBe('Plan one two');
		expect(clipboardData.getData('text/html')).toContain('<h1');
	});

	it('selects text in a rendered block on a drag and opens it on a click', async () => {
		const screen = render(MarkdownEditor, { value: 'Some rendered text' });
		const editor = editorIn(screen.container);
		const paragraph = () => screen.container.querySelector('.cm-md-block p');

		await vi.waitFor(() => expect(paragraph()?.textContent).toBe('Some rendered text'));

		const rect = paragraph()!.getBoundingClientRect();
		const y = rect.top + rect.height / 2;

		mouse('mousedown', paragraph()!, rect.left + 1, y);
		mouse('mousemove', paragraph()!, rect.left + rect.width / 2, y);
		mouse('mouseup', paragraph()!, rect.left + rect.width / 2, y);
		await settle();

		expect(document.getSelection()?.toString().length).toBeGreaterThan(0);
		expect(paragraph()).not.toBeNull();
		expect(editor.hasFocus).toBe(false);

		const clipboardData = new DataTransfer();

		paragraph()!.dispatchEvent(
			new ClipboardEvent('copy', { bubbles: true, cancelable: true, clipboardData })
		);

		expect(clipboardData.getData('text/plain')).toBe(document.getSelection()?.toString());

		mouse('mousedown', paragraph()!, rect.left + 1, y);
		mouse('mouseup', paragraph()!, rect.left + 1, y);

		// the focus change reaches the editor a moment later
		await vi.waitFor(() => {
			expect(screen.container.querySelector('.cm-md-block')).toBeNull();
			expect(editor.hasFocus).toBe(true);
		});
	});
});
