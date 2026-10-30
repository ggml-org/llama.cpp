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
	import { chatStore, conversationsStore, deviceStore } from '$lib/stores';
	import { untrack } from 'svelte';

	interface Props {
		/** Sizes and places the trigger, the inline dial size by default. */
		class?: string;
	}

	let { class: className = 'h-5 w-5 max-md:h-6 max-md:w-6' }: Props = $props();

	const gauge = useContextGauge();

	$effect(() => {
		const conv = conversationsStore.activeConversation;

		untrack(() => chatStore.processing.setActiveConversation(conv?.id ?? null));
	});

	$effect(() => {
		const conv = conversationsStore.activeConversation;
		const messages = conversationsStore.activeMessages as DatabaseMessage[];

		if (!conv) return;

		if (chatStore.isLoading || chatStore.isStreaming()) return;

		if (messages.length === 0) {
			untrack(() => chatStore.processing.setState(conv.id, null));

			return;
		}

		untrack(() => chatStore.processing.restoreFromMessages(messages, conv.id));
	});

	$effect(() => {
		gauge.startMonitoring();
	});
</script>

<div
	aria-label="Context usage"
	class={['flex cursor-default items-center justify-center', className]}
	data-context-gauge-trigger
	onclick={gaugeTriggerClick}
	onkeydown={gaugeTriggerKeydown}
	onpointerdown={gaugeTriggerPointerDown}
	onpointerenter={gaugeTriggerEnter}
	onpointerleave={gaugeTriggerLeave}
	role="button"
	tabindex="0"
>
	<!-- a phone gets the larger dial: the trigger is a touch target there -->
	<ContextGaugeDial
		level={gauge.colorLevel}
		percent={gauge.contextPercent}
		size={deviceStore.isMobile ? 'md' : 'sm'}
	/>
</div>
