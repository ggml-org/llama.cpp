// Guards the MarkdownEditor contract beyond the chat form: which edits reach onChange,
// undo of a programmatic replace, props changed after mount, blocks the live preview
// keeps raw or joins, the gap after a raw block, and selecting text in rendered blocks.

import { undo } from '@codemirror/commands';
import { EditorView } from '@codemirror/view';
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

	it('keeps a link reference definition raw', async () => {
		const screen = render(MarkdownEditor, { value: 'see [docs][d]\n\n[d]: https://example.com' });

		await vi.waitFor(() => {
			expect(screen.container.querySelectorAll('.cm-md-block')).toHaveLength(1);
			expect(screen.container.querySelector('.cm-md-raw-definition')?.textContent).toBe(
				'[d]: https://example.com'
			);
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
