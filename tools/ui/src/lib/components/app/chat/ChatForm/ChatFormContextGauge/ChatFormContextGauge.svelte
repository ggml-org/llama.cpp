<script lang="ts">
	import ContextGaugeDial from './ContextGaugeDial.svelte';
	import {
		gaugeTriggerClick,
		gaugeTriggerEnter,
		gaugeTriggerKeydown,
		gaugeTriggerLeave,
		gaugeTriggerPointerDown
	} from './gauge-popup.svelte';
	import { useContextGauge } from '$lib/hooks/use-context-gauge.svelte';
	import { chatStore, conversationsStore } from '$lib/stores';
	import { untrack } from 'svelte';

	const gauge = useContextGauge();

	$effect(() => {
		const conv = conversationsStore.activeConversation;

		untrack(() => chatStore.setActiveProcessingConversation(conv?.id ?? null));
	});

	$effect(() => {
		const conv = conversationsStore.activeConversation;
		const messages = conversationsStore.activeMessages as DatabaseMessage[];

		if (!conv) return;

		if (chatStore.isLoading || chatStore.isStreaming()) return;

		if (messages.length === 0) {
			untrack(() => chatStore.clearProcessingState(conv.id));

			return;
		}

		untrack(() => chatStore.restoreProcessingStateFromMessages(messages, conv.id));
	});

	$effect(() => {
		gauge.startMonitoring();
	});
</script>

<div
	role="button"
	tabindex="0"
	aria-label="Context usage"
	data-context-gauge-trigger
	class="flex h-5 w-5 cursor-default items-center justify-center"
	onclick={gaugeTriggerClick}
	onkeydown={gaugeTriggerKeydown}
	onpointerdown={gaugeTriggerPointerDown}
	onpointerenter={gaugeTriggerEnter}
	onpointerleave={gaugeTriggerLeave}
>
	<ContextGaugeDial percent={gauge.contextPercent} level={gauge.colorLevel} />
</div>
