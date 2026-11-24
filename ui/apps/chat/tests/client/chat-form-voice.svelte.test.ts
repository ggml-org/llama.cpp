// Guards voice input in the chat form, with a fake recorder in place of the microphone:
// what the microphone offers per model, the recording row, dictated text written into the
// message around what the user types, and a voice message whose transcript comes later.

import { modelServerProps, seedStores } from '../stories/chat/form/fixtures';
import { deferred, FakeVoiceRecorder } from '../stories/voice/fake-voice-recorder.svelte';
import ChatFormVoiceTestWrapper from './components/ChatFormVoiceTestWrapper.svelte';
import { EditorView } from '@codemirror/view';
import { VoiceTranscriptStatus } from '@ggml-org/llama-ui/voice';
import { VoiceRecorderStatus } from '@ggml-org/llama-voice';
import { SETTINGS_KEYS, VOICE_MODE_MESSAGE } from '#lib/constants/index.ts';
import { ServerRole, VoiceInputMode } from '#lib/enums/index.ts';
import { voiceTranscriptsStore } from '#lib/stores/index.ts';
import { parseFilesToMessageExtras } from '#lib/utils/browser-only.ts';
import { tick } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

let restore: (() => void) | undefined;

afterEach(() => {
	restore?.();
	restore = undefined;
});

function seed(
	audio: boolean,
	inputMode: VoiceInputMode = VoiceInputMode.AUTO,
	config: Record<string, unknown> = {}
) {
	restore = seedStores({
		config: {
			...config,
			autoMicOnEmpty: true,
			[SETTINGS_KEYS.VOICE_INPUT_MODE]: inputMode,
			[SETTINGS_KEYS.VOICE_TRANSCRIBE_MESSAGES]: true
		},
		serverProps: modelServerProps({ modalities: { audio, video: false, vision: false } }),
		serverRole: ServerRole.MODEL
	});
}

async function renderForm({ isLoading = false } = {}) {
	const recorder = new FakeVoiceRecorder();
	const state = $state({ files: [] as ChatUploadedFile[], sent: 0, value: '' });
	const screen = render(ChatFormVoiceTestWrapper, { isLoading, state, voiceRecorder: recorder });

	await tick();

	const content = screen.container.querySelector<HTMLElement>('.cm-content');
	const editor = content && EditorView.findFromDOM(content);

	if (!editor) throw new Error('editor not rendered');

	// an edit as typing makes it: reported to the form like a keystroke
	const type = (at: number, text: string) =>
		editor.dispatch({ changes: { from: at, insert: text }, userEvent: 'input.type' });
	const recordingRow = () => screen.container.querySelector('[data-slot="voice-recording-bar"]');

	return { editor, recorder, recordingRow, screen, state, type };
}

// the tooltip's trigger wraps the button and takes its name, so the button is the inner one
describe('the microphone', () => {
	it('sends a voice message to a model with audio input', async () => {
		seed(true);

		const { screen } = await renderForm();

		await expect
			.element(screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last())
			.toBeEnabled();
	});

	it('stands before the send button on an empty message, a quieter button than it', async () => {
		seed(true);

		const { screen } = await renderForm();
		// the send button keeps its place, waiting for a message
		const send = screen.getByRole('button', { exact: true, name: 'Send' });

		await expect.element(send).toBeDisabled();

		const mic = screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last();

		await expect.element(mic).toBeEnabled();
		expect(mic.element().getAttribute('data-variant')).toBe('tertiary');
		expect(
			mic.element().compareDocumentPosition(send.element()) & Node.DOCUMENT_POSITION_FOLLOWING
		).toBeTruthy();
	});

	it('dictates when the user always dictates', async () => {
		seed(true, VoiceInputMode.DICTATION);

		const { screen } = await renderForm();

		await expect
			.element(screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last())
			.toBeEnabled();
	});

	it('stays beside the send button once the message has text, and dictates more', async () => {
		seed(true, VoiceInputMode.DICTATION);

		const { recorder, screen, state, type } = await renderForm();

		type(0, 'Hello');
		await expect.poll(() => state.value).toBe('Hello');
		await expect.element(screen.getByRole('button', { name: 'Send' })).toBeEnabled();

		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last().click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		recorder.say('there');
		await expect.poll(() => state.value).toContain('there');
		expect(state.value).toContain('Hello');
	});

	it('stays after the stop button while a reply streams, looking like it', async () => {
		seed(true, VoiceInputMode.DICTATION);

		const { recorder, screen } = await renderForm({ isLoading: true });

		await expect.element(screen.getByRole('button', { exact: true, name: 'Stop' })).toBeEnabled();

		const mic = screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last();

		await expect.element(mic).toBeEnabled();

		// the last button of the row, filled like the stop button before it
		const stop = screen.getByRole('button', { exact: true, name: 'Stop' }).element();
		const micButton = mic.element();

		expect(stop.compareDocumentPosition(micButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(micButton.getAttribute('data-variant')).toBe(stop.getAttribute('data-variant'));

		await mic.click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);
	});

	it('leaves the send button on a single-model server without audio input', async () => {
		seed(false);

		const { screen } = await renderForm();

		await expect.element(screen.getByRole('button', { name: 'Send' })).toBeInTheDocument();
		expect(screen.container.innerHTML).not.toContain(VOICE_MODE_MESSAGE.voiceMessage);
	});

	it('does not stand beside the send button without voice input', async () => {
		seed(false);

		const { screen, state, type } = await renderForm();

		type(0, 'Hello');
		await expect.poll(() => state.value).toBe('Hello');
		await expect.element(screen.getByRole('button', { name: 'Send' })).toBeEnabled();
		expect(screen.container.innerHTML).not.toContain(VOICE_MODE_MESSAGE.voiceMessage);
		expect(screen.container.innerHTML).not.toContain(VOICE_MODE_MESSAGE.dictate);
	});
});

describe('a pointer on the microphone', () => {
	/** Lets the page use the microphone without asking, with a live stream as the device. */
	function grantMicrophone() {
		const context = new AudioContext();
		const stream = context.createMediaStreamDestination().stream;
		const query = vi
			.spyOn(navigator.permissions, 'query')
			.mockResolvedValue({ state: 'granted' } as PermissionStatus);
		const getUserMedia = vi.spyOn(navigator.mediaDevices, 'getUserMedia').mockResolvedValue(stream);
		const previous = restore;

		restore = () => {
			previous?.();
			query.mockRestore();
			getUserMedia.mockRestore();
			void context.close();
		};

		return { getUserMedia, stream };
	}

	it('opens the microphone ahead of the press, and the recording takes it', async () => {
		seed(true);

		const { getUserMedia, stream } = grantMicrophone();
		const { recorder, screen } = await renderForm();
		const mic = screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last();

		await mic.hover();
		await expect.poll(() => getUserMedia.mock.calls.length).toBe(1);

		await mic.click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		expect((recorder.options?.capture as { stream?: MediaStream }).stream).toBe(stream);
		// handed over, not asked for again
		expect(getUserMedia).toHaveBeenCalledOnce();
	});

	it('lets the microphone go when the pointer leaves without a press', async () => {
		seed(true);

		const { getUserMedia, stream } = grantMicrophone();
		const { screen } = await renderForm();

		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last().hover();
		await expect.poll(() => getUserMedia.mock.calls.length).toBe(1);

		await userEvent.unhover(
			screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last()
		);
		await expect.poll(() => stream.getAudioTracks()[0].readyState, { timeout: 4000 }).toBe('ended');
	});

	it('lets the microphone go when the recording is cancelled while it opens', async () => {
		seed(true);

		const { getUserMedia, stream } = grantMicrophone();
		const opening = deferred<MediaStream>();

		getUserMedia.mockReturnValueOnce(opening.promise);

		const { recorder, screen } = await renderForm();
		const mic = screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last();

		await mic.hover();
		await expect.poll(() => getUserMedia.mock.calls.length).toBe(1);
		await mic.click();
		await userEvent.keyboard('{Escape}');

		opening.resolve(stream);

		await expect.poll(() => stream.getAudioTracks()[0].readyState).toBe('ended');
		expect(recorder.status).toBe(VoiceRecorderStatus.IDLE);
	});
});

describe('while a reply streams', () => {
	it('offers to send what is dictated as soon as its first words are in the message', async () => {
		seed(true, VoiceInputMode.DICTATION);

		const { recorder, screen, state } = await renderForm({ isLoading: true });

		recorder.transcript = Promise.resolve('And one more thing');
		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last().click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);

		// nothing said yet: nothing to send
		expect(screen.container.innerHTML).not.toContain('Stop and send');

		recorder.say('And one more thing');
		await expect.poll(() => state.value).toBe('And one more thing');

		await screen.getByRole('button', { name: 'Stop and send' }).click();
		await expect.poll(() => state.sent).toBe(1);
	});

	it('offers to send a voice message once its live text shows', async () => {
		seed(true);

		const { recorder, screen } = await renderForm({ isLoading: true });

		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last().click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		expect(screen.container.innerHTML).not.toContain('Stop and send');

		recorder.say('What about tomorrow');
		await expect.element(screen.getByRole('button', { name: 'Stop and send' })).toBeInTheDocument();
	});

	it('keeps the microphone beside the send button once a voice message is attached', async () => {
		seed(true);

		const { recorder, screen, state } = await renderForm({ isLoading: true });

		recorder.transcript = Promise.resolve('What about tomorrow');
		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last().click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		recorder.say('What about tomorrow');
		await screen.getByRole('button', { name: 'Stop recording' }).last().click();
		await expect.poll(() => state.files.length).toBe(1);

		await expect.element(screen.getByRole('button', { exact: true, name: 'Send' })).toBeEnabled();
		await expect
			.element(screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last())
			.toBeEnabled();
	});
});

describe('recording', () => {
	it('asks for the raw microphone, and for the processing that is switched on', async () => {
		seed(true, VoiceInputMode.DICTATION);

		const raw = await renderForm();

		await raw.screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last().click();
		await expect.poll(() => raw.recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		expect(raw.recorder.options?.capture).not.toHaveProperty('constraints');
		raw.screen.unmount();
		restore?.();

		seed(true, VoiceInputMode.DICTATION, {
			[SETTINGS_KEYS.VOICE_ECHO_CANCELLATION]: true,
			[SETTINGS_KEYS.VOICE_NOISE_SUPPRESSION]: true
		});

		const filtered = await renderForm();

		await filtered.screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last().click();
		await expect.poll(() => filtered.recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		expect(filtered.recorder.options?.capture).toEqual({
			boost: true,
			constraints: { echoCancellation: true, noiseSuppression: true }
		});
	});

	it('boosts quiet speech unless the setting is off', async () => {
		seed(true, VoiceInputMode.DICTATION);

		const boosted = await renderForm();

		await boosted.screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last().click();
		await expect.poll(() => boosted.recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		expect(boosted.recorder.options?.capture).toMatchObject({ boost: true });
		boosted.screen.unmount();
		restore?.();

		seed(true, VoiceInputMode.DICTATION, { [SETTINGS_KEYS.VOICE_BOOST]: false });

		const plain = await renderForm();

		await plain.screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last().click();
		await expect.poll(() => plain.recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		expect(plain.recorder.options?.capture).toMatchObject({ boost: false });
	});

	it('says when the microphone is too quiet without moving the form, and stops once it is not', async () => {
		seed(true);

		const { recorder, screen } = await renderForm();

		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last().click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		expect(screen.container.innerHTML).not.toContain(VOICE_MODE_MESSAGE.tooQuiet);

		recorder.setQuiet(true);
		// inside the recording row, so the form keeps its height: read out, and shown above it
		await expect
			.element(screen.getByRole('status').filter({ hasText: VOICE_MODE_MESSAGE.tooQuiet }))
			.toBeInTheDocument();
		await expect
			.poll(() => document.querySelector('[data-slot="tooltip-content"]')?.textContent ?? '')
			.toContain(VOICE_MODE_MESSAGE.tooQuiet);

		recorder.setQuiet(false);
		await expect
			.poll(() => screen.container.querySelector('[data-slot="voice-recording-bar-warning"]'))
			.toBeNull();
	});

	it('shows the live waveform with cancel and stop, and Escape drops it', async () => {
		seed(true);

		const { recorder, recordingRow, screen } = await renderForm();

		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last().click();
		await expect.poll(recordingRow).toBeTruthy();

		recorder.setLevels(Array.from({ length: 48 }, (_, i) => (i % 5) / 5));
		await tick();

		expect(recordingRow()?.querySelector('[data-slot="voice-waveform-live"]')).toBeTruthy();
		await expect
			.element(screen.getByRole('button', { name: 'Cancel recording' }))
			.toBeInTheDocument();
		await expect
			.element(screen.getByRole('button', { name: 'Stop recording' }).last())
			.toBeEnabled();

		await userEvent.keyboard('{Escape}');
		await expect.poll(recordingRow).toBeNull();
		expect(recorder.cancelled).toBe(1);
	});
});

describe('dictation', () => {
	it('writes the live text into the message, keeping what is typed around it', async () => {
		seed(true, VoiceInputMode.DICTATION);

		const { recorder, screen, state, type } = await renderForm();
		const transcript = deferred<string>();

		recorder.transcript = transcript.promise;
		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last().click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);

		recorder.say('one');
		await expect.poll(() => state.value).toBe('one');

		// typed before and after the dictated text while it streams
		type(0, 'Note: ');
		await expect.poll(() => state.value).toBe('Note: one');
		type(state.value.length, ' (end)');
		recorder.say('one two');
		await expect.poll(() => state.value).toBe('Note: one two (end)');

		await screen.getByRole('button', { name: 'Stop recording' }).last().click();
		transcript.resolve('One, two.');

		await expect.poll(() => state.value).toBe('Note: One, two. (end)');
		expect(state.files).toHaveLength(0);
	});

	it('waits for the text after recording when none streamed, then inserts it', async () => {
		seed(true, VoiceInputMode.DICTATION);

		const { recorder, screen, state } = await renderForm();
		const transcript = deferred<string>();

		recorder.transcript = transcript.promise;
		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last().click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		await screen.getByRole('button', { name: 'Stop recording' }).last().click();

		await expect.element(screen.getByText(VOICE_MODE_MESSAGE.transcribing)).toBeInTheDocument();

		transcript.resolve('Hello there');
		await expect.poll(() => state.value).toBe('Hello there');
		expect(state.files).toHaveLength(0);
	});

	it('says the live text is being refined, and a send waits for the refined text', async () => {
		seed(true, VoiceInputMode.DICTATION);

		const { recorder, screen, state } = await renderForm();
		const transcript = deferred<string>();

		recorder.transcript = transcript.promise;
		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last().click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		recorder.say('I went. To the store.');
		await expect.poll(() => state.value).toBe('I went. To the store.');
		await screen.getByRole('button', { name: 'Stop recording' }).last().click();

		await expect.element(screen.getByText(VOICE_MODE_MESSAGE.refining)).toBeInTheDocument();

		await screen.getByRole('button', { name: 'Send' }).click();
		await expect
			.element(screen.getByText(VOICE_MODE_MESSAGE.sendAfterRefining))
			.toBeInTheDocument();
		expect(state.sent).toBe(0);

		transcript.resolve('I went to the store.');
		await expect.poll(() => state.sent).toBe(1);
		expect(state.value).toBe('I went to the store.');
		expect(screen.container.querySelector('[data-slot="voice-refining"]')).toBeNull();
	});

	it('stops and sends the dictated text in one go', async () => {
		seed(true, VoiceInputMode.DICTATION);

		const { recorder, screen, state } = await renderForm();

		recorder.transcript = Promise.resolve('Hello there');
		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last().click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		recorder.say('Hello there');

		await screen.getByRole('button', { name: 'Stop and send' }).click();
		await expect.poll(() => state.sent).toBe(1);
		expect(state.value).toBe('Hello there');
	});

	it('takes the dictated text back out on cancel', async () => {
		seed(true, VoiceInputMode.DICTATION);

		const { recorder, screen, state } = await renderForm();

		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.dictate }).last().click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		recorder.say('never mind');
		await expect.poll(() => state.value).toBe('never mind');

		await screen.getByRole('button', { name: 'Cancel recording' }).click();
		await expect.poll(() => state.value).toBe('');
	});
});

describe('a voice message', () => {
	it('turns its live text into text to edit on focus, which goes with the clip on stop', async () => {
		seed(true);

		const { editor, recorder, screen, state } = await renderForm();

		recorder.transcript = Promise.resolve('model transcript');
		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last().click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);
		recorder.say('Hello wrold');

		editor.focus();
		await expect.poll(() => state.value).toBe('Hello wrold');

		// fixed by hand, and what is said next follows it
		editor.dispatch({ changes: { from: 7, insert: 'or', to: 9 }, userEvent: 'input.type' });
		await expect.poll(() => state.value).toBe('Hello world');
		recorder.say('Hello wrold, how are you');
		await expect.poll(() => state.value).toBe('Hello world, how are you');

		await screen.getByRole('button', { name: 'Stop recording' }).last().click();
		await expect.poll(() => state.files.length).toBe(1);
		expect(state.value).toBe('');
		expect(voiceTranscriptsStore.transcriptOf(state.files[0].name)).toBe(
			'Hello world, how are you'
		);
	});

	it('is attached as a voice clip that gets its transcript later', async () => {
		seed(true);

		const { recorder, screen, state } = await renderForm();
		const transcript = deferred<string>();

		recorder.transcript = transcript.promise;
		await screen.getByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage }).last().click();
		await expect.poll(() => recorder.status).toBe(VoiceRecorderStatus.RECORDING);

		// streamed text shows in the empty message box as its placeholder, not in the message
		const placeholder = () => screen.container.querySelector('.cm-placeholder')?.textContent;

		recorder.say('What is the weather on Mars');
		await expect.poll(placeholder).toBe('What is the weather on Mars');
		expect(state.value).toBe('');

		await screen.getByRole('button', { name: 'Stop recording' }).last().click();
		await expect.poll(() => state.files.length).toBe(1);
		// the text now goes with the clip
		await expect.poll(placeholder).not.toBe('What is the weather on Mars');

		const [file] = state.files;

		expect(file.voice).toMatchObject({ durationMs: 2400 });
		expect(file.voice?.peaks).toHaveLength(64);
		expect(state.value).toBe('');
		expect(voiceTranscriptsStore.get(file.name)?.status).toBe(VoiceTranscriptStatus.TRANSCRIBING);

		// drawn as a voice chip, its transcript on the way
		const reveal = () =>
			screen.container.querySelector('[data-slot="chat-attachment-voice"] [data-status]');

		await expect
			.poll(() => reveal()?.getAttribute('data-status'))
			.toBe(VoiceTranscriptStatus.TRANSCRIBING);

		transcript.resolve('What is the weather on Mars?');
		await expect
			.poll(() => voiceTranscriptsStore.transcriptOf(file.name))
			.toBe('What is the weather on Mars?');

		await expect.poll(() => reveal()?.getAttribute('data-status')).toBe(VoiceTranscriptStatus.DONE);

		const { extras } = await parseFilesToMessageExtras($state.snapshot(state.files));

		expect(extras[0]).toMatchObject({
			durationMs: 2400,
			transcript: 'What is the weather on Mars?',
			voice: true
		});

		voiceTranscriptsStore.forget(file.name);
	});
});
