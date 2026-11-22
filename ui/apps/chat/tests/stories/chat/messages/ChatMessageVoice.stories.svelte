<script lang="ts" module>
	import { demoStoredClip } from '../../voice/fixtures';
	import { CONV_ID, noopChatActions, seedChatStory, STORY_CLASS } from './fixtures';
	import { VoiceTranscriptStatus } from '@ggml-org/llama-ui/voice';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessage from '#lib/components/app/chat/ChatMessages/ChatMessage/ChatMessage.svelte';
	import { SETTINGS_KEYS } from '#lib/constants/index.ts';
	import { AttachmentType, MessageRole } from '#lib/enums/index.ts';
	import { voiceTranscriptsStore } from '#lib/stores/index.ts';
	import type { DatabaseMessageExtraAudioFile, SettingsConfigValue } from '#lib/types/index.ts';
	import { expect, waitFor } from 'storybook/test';

	const { Story } = defineMeta({
		component: ChatMessage,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Chat/ChatMessages/ChatMessageVoice'
	});

	const TRANSCRIPT =
		'Can you explain how speculative decoding speeds up generation, and when it does not help?';

	// filled in before each story: the clip is made in the browser
	let message = $state<DatabaseMessage>({
		children: [],
		content: '',
		convId: CONV_ID,
		extra: [],
		id: 'user-voice',
		parent: null,
		role: MessageRole.USER,
		timestamp: Date.now(),
		type: 'text'
	});

	async function seedVoice(
		clip: Partial<DatabaseMessageExtraAudioFile>,
		content = '',
		settings: Record<string, SettingsConfigValue> = {}
	) {
		const stored = await demoStoredClip(7, 3);
		const name = clip.name ?? 'voice-story.wav';

		message.content = content;
		message.extra = [
			{
				base64Data: stored.base64,
				durationMs: stored.durationMs,
				mimeType: 'audio/wav',
				name,
				peaks: stored.peaks,
				size: stored.size,
				type: AttachmentType.AUDIO,
				voice: true,
				...clip
			} as DatabaseMessageExtraAudioFile
		];

		const restore = seedChatStory({ settings });

		return () => {
			restore();
			voiceTranscriptsStore.forget(name);
		};
	}
</script>

<!-- the transcript is open under the clip, and the toggle beside it folds it -->
<Story
	asChild
	beforeEach={() => seedVoice({ transcript: TRANSCRIPT })}
	name="VoiceMessage"
	play={async ({ canvas, canvasElement }) => {
		const reveal = await waitFor(() => {
			const element = canvasElement.querySelector('[data-slot="voice-transcript-reveal"]');

			if (!element) throw new Error('no voice message');

			return element as HTMLElement;
		});

		await expect(reveal).toHaveAttribute('data-revealed');
		// drawn word by word, see VoiceSpokenText
		await expect(reveal).toHaveTextContent(TRANSCRIPT);
		await expect(canvas.getByRole('button', { name: 'Hide transcript' })).toBeInTheDocument();
	}}
>
	<ChatMessage chatActions={noopChatActions} class={STORY_CLASS} {message} />
</Story>

<!-- with Collapse transcripts on: the transcript opens under the clip on hover, keyboard focus or a tap -->
<Story
	asChild
	beforeEach={() =>
		seedVoice({ transcript: TRANSCRIPT }, '', { [SETTINGS_KEYS.VOICE_COLLAPSE_TRANSCRIPTS]: true })}
	name="VoiceMessageCollapsed"
	play={async ({ canvasElement, userEvent }) => {
		const reveal = await waitFor(() => {
			const element = canvasElement.querySelector('[data-slot="voice-transcript-reveal"]');

			if (!element) throw new Error('no voice message');

			return element as HTMLElement;
		});

		await expect(reveal).not.toHaveAttribute('data-revealed');
		await userEvent.hover(reveal);
		await waitFor(() => expect(reveal).toHaveAttribute('data-revealed'));
		// drawn word by word, see VoiceSpokenText
		await expect(reveal).toHaveTextContent(TRANSCRIPT);
	}}
>
	<ChatMessage chatActions={noopChatActions} class={STORY_CLASS} {message} />
</Story>

<Story
	asChild
	beforeEach={() => seedVoice({ transcript: TRANSCRIPT }, 'Answer in two sentences.')}
	name="VoiceMessageWithText"
>
	<ChatMessage chatActions={noopChatActions} class={STORY_CLASS} {message} />
</Story>

<!-- sent before its transcript came back -->
<Story
	asChild
	beforeEach={async () => {
		const restore = await seedVoice({ name: 'voice-pending.wav' });

		voiceTranscriptsStore.track('voice-pending.wav', new Promise<string>(() => {}));

		return restore;
	}}
	name="VoiceMessageTranscribing"
	play={async ({ canvasElement }) => {
		await waitFor(() =>
			expect(canvasElement.querySelector('[data-slot="voice-transcript-reveal"]')).toHaveAttribute(
				'data-status',
				VoiceTranscriptStatus.TRANSCRIBING
			)
		);
	}}
>
	<ChatMessage chatActions={noopChatActions} class={STORY_CLASS} {message} />
</Story>

<!-- stored before voice messages kept a waveform: the peaks come from the audio -->
<Story
	asChild
	beforeEach={() => seedVoice({ peaks: undefined, transcript: TRANSCRIPT })}
	name="VoiceMessageWithoutPeaks"
	play={async ({ canvasElement }) => {
		await waitFor(() =>
			expect(
				canvasElement.querySelectorAll('[data-slot="voice-waveform"] [aria-hidden] span').length
			).toBeGreaterThan(0)
		);
	}}
>
	<ChatMessage chatActions={noopChatActions} class={STORY_CLASS} {message} />
</Story>
