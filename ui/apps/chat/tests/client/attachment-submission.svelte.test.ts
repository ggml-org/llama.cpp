import Harness from './components/AttachmentSubmissionHarness.svelte';
import { EditorView } from '@codemirror/view';
import { AttachmentType, MessageRole, MessageType, ServerRole } from '#lib/enums/index.ts';
import {
	chatStore,
	conversationsStore,
	draftMessagesStore,
	modelsStore,
	serverStore
} from '#lib/stores/index.ts';
import type {
	ApiLlamaCppServerProps,
	ChatMessageActions,
	DatabaseMessage,
	DatabaseMessageExtra
} from '#lib/types/index.ts';
import { tick } from 'svelte';
import { toast } from 'svelte-sonner';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';
import { cleanup, render } from 'vitest-browser-svelte';

const existingExtra: DatabaseMessageExtra = {
	content: 'existing attachment',
	name: 'existing.txt',
	type: AttachmentType.TEXT
};

function editor(container: HTMLElement) {
	const element = container.querySelector<HTMLElement>('.cm-content');
	const view = element && EditorView.findFromDOM(element);

	if (!view) throw new Error('message editor missing');

	return view;
}

async function writeDraft(container: HTMLElement, text: string) {
	const view = editor(container);

	view.dispatch({ changes: { from: 0, insert: text, to: view.state.doc.length } });
	await tick();
}

function actions(): ChatMessageActions {
	return {
		continueAssistantMessage: vi.fn(),
		copy: vi.fn(),
		delete: vi.fn(),
		editUserMessagePreserveResponses: vi.fn(),
		editWithBranching: vi.fn(),
		editWithReplacement: vi.fn(),
		forkConversation: vi.fn(),
		navigateToSibling: vi.fn(),
		regenerateCompaction: vi.fn(),
		regenerateWithBranching: vi.fn()
	};
}

async function upload(
	container: HTMLElement,
	name: string,
	type = 'audio/wav',
	bytes = new Uint8Array([1, 2, 3])
) {
	const input = container.querySelector<HTMLInputElement>('input[type="file"]');

	if (!input) throw new Error('file input missing');

	const files = new DataTransfer();

	files.items.add(new File([new Uint8Array(bytes)], name, { type }));
	input.files = files.files;
	input.dispatchEvent(new Event('change', { bubbles: true }));
	await expect.poll(() => container.textContent).toContain(name);
	await expect.element(page.getByRole('button', { exact: true, name: 'Send' })).toBeEnabled();
}

function failRead(name: string) {
	const read = FileReader.prototype.readAsDataURL;

	return vi.spyOn(FileReader.prototype, 'readAsDataURL').mockImplementation(function (
		this: FileReader,
		file: Blob
	) {
		if (file instanceof File && file.name === name) {
			Object.defineProperty(this, 'error', {
				value: new DOMException('File unavailable', 'NotReadableError')
			});
			queueMicrotask(() => this.dispatchEvent(new ProgressEvent('error')));

			return;
		}

		read.call(this, file);
	});
}

function holdRead(name: string) {
	const nativeRead = FileReader.prototype.readAsDataURL;

	let held: { reader: FileReader; file: File } | undefined;

	const spy = vi.spyOn(FileReader.prototype, 'readAsDataURL').mockImplementation(function (
		this: FileReader,
		file: Blob
	) {
		if (file instanceof File && file.name === name) {
			held = { file, reader: this };

			return;
		}

		nativeRead.call(this, file);
	});

	return {
		async release(success = true) {
			if (!held) throw new Error('attachment read did not start');

			const { file, reader } = held;

			if (success) {
				const done = new Promise<void>((resolve) =>
					reader.addEventListener('loadend', () => resolve(), { once: true })
				);

				nativeRead.call(reader, file);
				await done;
			} else {
				Object.defineProperty(reader, 'error', {
					value: new DOMException('File unavailable', 'NotReadableError')
				});
				reader.dispatchEvent(new ProgressEvent('error'));
			}
		},
		spy
	};
}

describe('attachment failure at submission', () => {
	beforeEach(() => {
		serverStore.clear();
		serverStore.role = ServerRole.MODEL;
		serverStore.props = {
			chat_template: '',
			default_generation_settings: { n_ctx: 2048 },
			modalities: { audio: true, video: true, vision: true },
			model_path: 'test-model.gguf'
		} as ApiLlamaCppServerProps;
		modelsStore.models = [
			{ capabilities: [], id: 'test-model.gguf', model: 'test-model.gguf', name: 'Test model' }
		];
		modelsStore.clearSelection();
		conversationsStore.activeConversation = null;
		conversationsStore.activeMessages = [];
		chatStore.clearEditMode();
		chatStore.clearPendingEditMessageId();
		draftMessagesStore.clearDraftMessage(undefined);
	});
	afterEach(() => {
		cleanup();
		toast.dismiss();
		vi.restoreAllMocks();
	});

	it('restores the composer and retries with the same files', async () => {
		const send = vi.spyOn(chatStore, 'sendMessage').mockResolvedValue();
		const { container } = render(Harness, { surface: 'screen' });

		await writeDraft(container, 'describe the recording');
		await upload(container, 'composer.wav');
		const failure = failRead('composer.wav');

		await page.getByRole('button', { exact: true, name: 'Send' }).click();
		await expect
			.element(page.getByText('Could not read: composer.wav. Retry or remove these files.'))
			.toBeVisible();
		expect(send).not.toHaveBeenCalled();
		await expect.poll(() => editor(container).state.doc.toString()).toBe('describe the recording');
		expect(container.textContent).toContain('composer.wav');

		failure.mockRestore();
		await page.getByRole('button', { exact: true, name: 'Send' }).click();
		await expect.poll(() => send.mock.calls.length).toBe(1);
		expect(send.mock.calls[0][0]).toBe('describe the recording');
		expect(send.mock.calls[0][1]).toEqual([
			expect.objectContaining({ base64Data: 'AQID', mimeType: 'audio/wav', name: 'composer.wav' })
		]);
		await expect.poll(() => editor(container).state.doc.toString()).toBe('');
	});

	it.each([false, true])(
		'preserves saved editing and the save-only choice (%s)',
		async (saveOnly) => {
			const chatActions = actions();
			const message: DatabaseMessage = {
				children: [],
				content: 'original prompt',
				convId: 'conversation-1',
				extra: [existingExtra],
				id: 'user-1',
				parent: null,
				role: MessageRole.USER,
				timestamp: 1,
				type: MessageType.TEXT
			};
			const { container } = render(Harness, { chatActions, message, surface: 'saved' });

			await page.getByRole('button', { exact: true, name: 'Edit' }).click();
			await writeDraft(container, 'revised prompt');
			const name = saveOnly ? 'save-only.wav' : 'regenerate.wav';

			await upload(container, name);
			const choice = page.getByRole('switch', { name: 'Update without re-sending' });

			if (saveOnly) await choice.click();

			const failure = failRead(name);

			await page.getByRole('button', { exact: true, name: 'Send' }).click();
			await expect
				.element(page.getByText(`Could not read: ${name}. Retry or remove these files.`))
				.toBeVisible();
			expect(chatActions.editWithBranching).not.toHaveBeenCalled();
			expect(chatActions.editUserMessagePreserveResponses).not.toHaveBeenCalled();
			await expect.poll(() => editor(container).state.doc.toString()).toBe('revised prompt');
			expect(container.textContent).toContain(name);

			if (saveOnly) await expect.element(choice).toBeChecked();

			failure.mockRestore();
			await page.getByRole('button', { exact: true, name: 'Send' }).click();
			const selected = vi.mocked(
				saveOnly ? chatActions.editUserMessagePreserveResponses : chatActions.editWithBranching
			);

			await expect.poll(() => selected.mock.calls.length).toBe(1);
			expect(selected.mock.calls[0][2]?.map((extra) => extra.name)).toEqual(['existing.txt', name]);
			expect(selected.mock.calls[0][2]?.[1]).toMatchObject({
				base64Data: 'AQID',
				mimeType: 'audio/wav'
			});
			await expect.poll(() => container.querySelector('.cm-content')).toBeNull();

			if (saveOnly) {
				expect(chatActions.editWithBranching).not.toHaveBeenCalled();
				await page.getByRole('button', { exact: true, name: 'Edit' }).click();
				await expect
					.element(page.getByRole('switch', { name: 'Update without re-sending' }))
					.not.toBeChecked();
			}
		}
	);

	it('keeps a pending edit until all attachments can be saved', async () => {
		const onEdit = vi.fn();
		const { container } = render(Harness, {
			content: 'pending prompt',
			extras: [existingExtra],
			onEdit,
			surface: 'pending'
		});

		await page.getByRole('button', { exact: true, name: 'Edit' }).click();
		await writeDraft(container, 'revised pending prompt');
		await upload(container, 'pending.wav');
		const failure = failRead('pending.wav');

		await page.getByRole('button', { exact: true, name: 'Send' }).click();
		await expect
			.element(page.getByText('Could not read: pending.wav. Retry or remove these files.'))
			.toBeVisible();
		expect(onEdit).not.toHaveBeenCalled();
		await expect.poll(() => editor(container).state.doc.toString()).toBe('revised pending prompt');
		expect(container.textContent).toContain('pending.wav');

		failure.mockRestore();
		await page.getByRole('button', { exact: true, name: 'Send' }).click();
		await expect.poll(() => onEdit.mock.calls.length).toBe(1);
		expect(onEdit.mock.calls[0][0]).toBe('revised pending prompt');
		expect(onEdit.mock.calls[0][1].map((extra: DatabaseMessageExtra) => extra.name)).toEqual([
			'existing.txt',
			'pending.wav'
		]);
		await expect.poll(() => container.querySelector('.cm-content')).toBeNull();
	});
	it.each([false, true])(
		'preserves edits made during a delayed attachment read (success=%s)',
		async (succeeds) => {
			const send = vi.spyOn(chatStore, 'sendMessage').mockResolvedValue();
			const { container } = render(Harness, { surface: 'screen' });

			await writeDraft(container, 'first draft');
			const name = succeeds ? 'delayed-success.wav' : 'delayed-error.wav';

			await upload(container, name);
			const read = holdRead(name);

			await page.getByRole('button', { exact: true, name: 'Send' }).click();
			await expect.poll(() => read.spy.mock.calls.length).toBe(1);
			await page.getByRole('button', { exact: true, name: 'Send' }).click();
			expect(read.spy.mock.calls).toHaveLength(1);
			await writeDraft(container, 'newer draft');
			await upload(container, 'newer.wav');
			await read.release(succeeds);

			if (succeeds) {
				await expect
					.element(page.getByText('Message changed while preparing attachments. Send again.'))
					.toBeVisible();
			} else {
				await expect
					.element(page.getByText(`Could not read: ${name}. Retry or remove these files.`))
					.toBeVisible();
			}

			expect(send).not.toHaveBeenCalled();
			await expect.poll(() => editor(container).state.doc.toString()).toBe('newer draft');
			expect(container.textContent).toContain(name);
			expect(container.textContent).toContain('newer.wav');
			read.spy.mockRestore();
			await page.getByRole('button', { exact: true, name: 'Send' }).click();
			await expect.poll(() => send.mock.calls.length).toBe(1);
			expect(send.mock.calls[0][0]).toBe('newer draft');
			expect(send.mock.calls[0][1]?.map((extra) => extra.name)).toEqual([name, 'newer.wav']);
		}
	);
	it('keeps a draft when a message editor opens during an attachment read', async () => {
		const send = vi.spyOn(chatStore, 'sendMessage').mockResolvedValue();
		const { container } = render(Harness, { surface: 'screen' });

		await writeDraft(container, 'keep this draft');
		await upload(container, 'paused.wav');
		const read = holdRead('paused.wav');

		await page.getByRole('button', { exact: true, name: 'Send' }).click();
		await expect.poll(() => read.spy.mock.calls.length).toBe(1);
		chatStore.setEditModeActive(vi.fn());
		await tick();
		await read.release();
		expect(send).not.toHaveBeenCalled();
		expect(editor(container).state.doc.toString()).toBe('keep this draft');
		expect(container.textContent).toContain('paused.wav');

		chatStore.clearEditMode();
		read.spy.mockRestore();
		await page.getByRole('button', { exact: true, name: 'Send' }).click();
		await expect.poll(() => send.mock.calls.length).toBe(1);
	});

	it.each([
		{ cancel: false, failed: false },
		{ cancel: true, failed: false },
		{ cancel: true, failed: true }
	])(
		'discards a pending edit read after the editor changes (cancel=$cancel, failed=$failed)',
		async ({ cancel, failed }) => {
			const report = vi.spyOn(toast, 'error');
			const onEdit = vi.fn();
			const { container } = render(Harness, {
				content: 'queued draft',
				extras: [],
				onEdit,
				surface: 'pending'
			});

			await page.getByRole('button', { exact: true, name: 'Edit' }).click();
			await writeDraft(container, 'first edit');
			await upload(container, 'edit.wav');
			const read = holdRead('edit.wav');

			await page.getByRole('button', { exact: true, name: 'Send' }).click();
			await expect.poll(() => read.spy.mock.calls.length).toBe(1);

			if (cancel) {
				await page.getByRole('button', { exact: true, name: 'Cancel' }).click();
				await page.getByRole('button', { exact: true, name: 'Discard' }).click();
				await expect.poll(() => container.querySelector('.cm-content')).toBeNull();
			} else {
				await writeDraft(container, 'updated edit');
			}

			await read.release(!failed);
			await tick();
			expect(onEdit).not.toHaveBeenCalled();

			if (cancel) {
				expect(report).not.toHaveBeenCalled();
			} else {
				await expect
					.element(page.getByText('Edit changed while preparing attachments. Save again.'))
					.toBeVisible();
				await expect.poll(() => editor(container).state.doc.toString()).toBe('updated edit');
				read.spy.mockRestore();
				await page.getByRole('button', { exact: true, name: 'Send' }).click();
				await expect.poll(() => onEdit.mock.calls.length).toBe(1);
				expect(onEdit.mock.calls[0][0]).toBe('updated edit');
			}
		}
	);

	it('keeps the prompt and healthy files when removing an empty attachment', async () => {
		const send = vi.spyOn(chatStore, 'sendMessage').mockResolvedValue();
		const { container } = render(Harness, { surface: 'screen' });

		await writeDraft(container, 'keep this prompt');
		await upload(container, 'healthy.wav');
		await upload(container, 'empty.txt', 'text/plain', new Uint8Array());
		await page.getByRole('button', { exact: true, name: 'Send' }).click();
		await expect
			.element(page.getByRole('alertdialog', { name: 'Empty Files Detected' }))
			.toBeVisible();
		await page.getByRole('button', { exact: true, name: 'Got it' }).click();
		expect(send).not.toHaveBeenCalled();
		await expect.poll(() => editor(container).state.doc.toString()).toBe('keep this prompt');
		expect(container.textContent).toContain('healthy.wav');
		expect(container.textContent).not.toContain('empty.txt');
	});
});
