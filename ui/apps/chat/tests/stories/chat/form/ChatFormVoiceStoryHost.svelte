<!--
	The chat form with a fake microphone: a press on the microphone records levels like someone speaking and, with a
	script, streams it as a realtime model would. Voice messages it records appear as chips.
-->
<script lang="ts">
	import { DemoVoiceRecorder } from '../../voice/fake-voice-recorder.svelte';
	import { demoLevel } from '../../voice/fixtures';
	import ChatForm from '#lib/components/app/chat/ChatForm/ChatForm.svelte';
	import { untrack } from 'svelte';

	interface Props {
		class?: string;
		/** What the speaker says, streamed word by word; empty streams nothing. */
		script?: string;
		/** The message the form starts with. */
		value?: string;
		uploadedFiles?: ChatUploadedFile[];
	}

	let {
		class: className = '',
		script = '',
		uploadedFiles = $bindable([]),
		value = $bindable('')
	}: Props = $props();

	const recorder = new DemoVoiceRecorder(
		untrack(() => script),
		(t) => demoLevel(t)
	);
</script>

<ChatForm bind:uploadedFiles bind:value class={className} voiceRecorder={recorder} />
