<script lang="ts">
	import ContextGaugeDial from './ContextGaugeDial.svelte';
	import {
		gaugeTriggerClick,
		gaugeTriggerEnter,
		gaugeTriggerKeydown,
		gaugeTriggerLeave,
		gaugeTriggerPointerDown
	} from './gauge-popup.svelte';
	import GlassKey from '#lib/components/app/actions/GlassKey.svelte';
	import { useContextGauge } from '#lib/hooks/use-context-gauge.svelte.ts';
	import { chatStore, conversationsStore, deviceStore } from '#lib/stores/index.ts';
	import { untrack } from 'svelte';

	interface Props {
		/** Sizes and places the trigger, the inline dial size by default. */
		class?: string;
		/** Draws the dial as the icon of a phone's corner key, matching the menu key's glyph. */
		corner?: boolean;
	}

	let { class: className = 'h-5 w-5 max-md:h-6 max-md:w-6', corner = false }: Props = $props();

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

{#snippet dial(size: 'sm' | 'md' | 'key')}
	<ContextGaugeDial level={gauge.colorLevel} percent={gauge.contextPercent} {size} />
{/snippet}

{#if corner}
	<!-- a phone's corner key, the same glass key as the menu key across from it -->
	<GlassKey
		class={className}
		data-context-gauge-trigger
		label="Context usage"
		onclick={gaugeTriggerClick}
		onkeydown={gaugeTriggerKeydown}
		onpointerdown={gaugeTriggerPointerDown}
		onpointerenter={gaugeTriggerEnter}
		onpointerleave={gaugeTriggerLeave}
	>
		{@render dial('key')}
	</GlassKey>
{:else}
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
		{@render dial(deviceStore.isMobile ? 'md' : 'sm')}
	</div>
{/if}
