<script lang="ts" module>
	import { demoClip } from '../../voice/fixtures';
	import ChatFormVoiceStoryHost from './ChatFormVoiceStoryHost.svelte';
	import {
		FORM_CLASS,
		modelServerProps,
		resetPointerEvents,
		seedStores,
		ServerRole
	} from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { SETTINGS_KEYS, VOICE_MODE_MESSAGE } from '#lib/constants/index.ts';
	import { VoiceInputMode } from '#lib/enums/index.ts';
	import { expect, waitFor } from 'storybook/test';

	const { Story } = defineMeta({
		beforeEach: resetPointerEvents,
		component: ChatFormVoiceStoryHost,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Chat/ChatForm/Voice'
	});

	const SCRIPT = 'What is the difference between a KV cache and a prompt cache';

	function audioServer(inputMode: VoiceInputMode) {
		return seedStores({
			config: {
				autoMicOnEmpty: true,
				[SETTINGS_KEYS.VOICE_INPUT_MODE]: inputMode,
				[SETTINGS_KEYS.VOICE_TRANSCRIBE_MESSAGES]: true
			},
			serverProps: modelServerProps({ modalities: { audio: true, video: false, vision: true } }),
			serverRole: ServerRole.MODEL
		});
	}

	function voiceChip(): ChatUploadedFile {
		const clip = demoClip(6, 2);

		return {
			file: new File([], 'voice-demo.wav', { type: 'audio/wav' }),
			id: 'voice-demo',
			name: 'voice-demo.wav',
			preview: clip.src,
			size: 96000,
			type: 'audio/wav',
			voice: { durationMs: clip.durationMs, peaks: clip.peaks }
		};
	}
</script>

<!-- a model with audio input: the recording becomes a voice message, its live text under the waveform -->
<Story
	args={{ class: FORM_CLASS, script: SCRIPT }}
	beforeEach={() => audioServer(VoiceInputMode.AUTO)}
	name="RecordingVoiceMessage"
	play={async ({ canvas, canvasElement, userEvent }) => {
		const mic = canvas.getAllByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage });

		await userEvent.click(mic[mic.length - 1]);

		await expect(await canvas.findByText(VOICE_MODE_MESSAGE.recording)).toBeInTheDocument();
		await expect(canvas.getByRole('button', { name: 'Cancel recording' })).toBeInTheDocument();
		// the live text shows as the empty message box's placeholder, the message stays empty
		await waitFor(
			() => expect(canvasElement.querySelector('.cm-placeholder')).toHaveTextContent(/What is the/),
			{ timeout: 3000 }
		);
		// a placeholder shows only while the message itself is empty
	}}
/>

<!-- dictation: the text appears in the message as it is spoken -->
<Story
	args={{ class: FORM_CLASS, script: SCRIPT }}
	beforeEach={() => audioServer(VoiceInputMode.DICTATION)}
	name="RecordingDictation"
	play={async ({ canvas, userEvent }) => {
		const mic = canvas.getAllByRole('button', { name: VOICE_MODE_MESSAGE.dictate });

		await userEvent.click(mic[mic.length - 1]);

		await expect(await canvas.findByText(VOICE_MODE_MESSAGE.dictating)).toBeInTheDocument();
		await waitFor(() => expect(canvas.getByRole('textbox')).toHaveTextContent('What is the'), {
			timeout: 3000
		});
	}}
/>

<!-- stop records the voice message as a chip in the form, its transcript on the way -->
<Story
	args={{ class: FORM_CLASS, script: SCRIPT }}
	beforeEach={() => audioServer(VoiceInputMode.AUTO)}
	name="VoiceMessageRecorded"
	play={async ({ canvas, userEvent }) => {
		const mic = canvas.getAllByRole('button', { name: VOICE_MODE_MESSAGE.voiceMessage });

		await userEvent.click(mic[mic.length - 1]);
		await canvas.findByText(VOICE_MODE_MESSAGE.recording);
		await new Promise((resolve) => setTimeout(resolve, 1200));

		const stop = canvas.getAllByRole('button', { name: 'Stop recording' });

		await userEvent.click(stop[stop.length - 1]);

		await waitFor(() =>
			expect(canvas.getAllByRole('slider', { name: /Voice message/ }).length).toBeGreaterThan(0)
		);
	}}
/>

<Story
	args={{ class: FORM_CLASS, uploadedFiles: [voiceChip()] }}
	beforeEach={() => audioServer(VoiceInputMode.AUTO)}
	name="WithVoiceMessage"
/>
