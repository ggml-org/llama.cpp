<!--
	The live waveform in a chat form row, fed by a fake microphone meter: a level every 50 ms while `active`, and the
	recording time beside it.
-->
<script lang="ts">
	import { demoLevel } from './fixtures';
	import { formatClipTime, VoiceWaveformLive } from '@ggml-org/llama-ui/voice';
	import ChatFormActionRecord from '#lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionRecord.svelte';
	import { onMount } from 'svelte';

	interface Props {
		active?: boolean;
		class?: string;
		/** Levels to start from, e.g. a paused recording's. */
		initialSeconds?: number;
		seed?: number;
	}

	let { active = true, class: className = '', initialSeconds = 0, seed = 1 }: Props = $props();

	const TICK_MS = 50;
	const KEEP = 160;

	let elapsedMs = $state(0);
	let levels = $state<number[]>([]);

	onMount(() => {
		elapsedMs = initialSeconds * 1000;
		levels = Array.from({ length: Math.min(Math.round(initialSeconds * 20), KEEP) }, (_, index) =>
			demoLevel(index / 20, seed)
		);

		const timer = setInterval(() => {
			if (!active) return;

			elapsedMs += TICK_MS;
			levels = [...levels.slice(-(KEEP - 1)), demoLevel(elapsedMs / 1000, seed)];
		}, TICK_MS);

		return () => clearInterval(timer);
	});
</script>

<div
	class="chat-form-surface flex items-center gap-2 rounded-(--chat-form-single-row-radius) px-3 py-2.5 {className}"
>
	<VoiceWaveformLive {active} class="flex-1 px-2" label="Recording" {levels} />

	<span class="w-9 shrink-0 text-right text-caption text-muted-foreground tabular-nums">
		{formatClipTime(elapsedMs)}
	</span>

	<ChatFormActionRecord isRecording={active} />
</div>
