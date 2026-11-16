<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ChatForm from '#lib/components/app/chat/ChatForm/ChatForm.svelte';
	import type { VoiceRecorderLike } from '#lib/hooks/use-chat-form-voice.svelte.ts';

	interface Props {
		/** The form's message and files, kept in step with it for the test to read, and how often it sent. */
		state: { files: ChatUploadedFile[]; sent?: number; value: string };
		voiceRecorder: VoiceRecorderLike;
	}

	let { state, voiceRecorder }: Props = $props();
</script>

<Tooltip.Provider>
	<ChatForm
		bind:uploadedFiles={state.files}
		bind:value={state.value}
		onSubmit={() => (state.sent = (state.sent ?? 0) + 1)}
		{voiceRecorder}
	/>
</Tooltip.Provider>
