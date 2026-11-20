<script lang="ts">
	import ContextGaugeDial from './ContextGaugeDial.svelte';
	import ContextGaugePanel from './ContextGaugePanel.svelte';
	import { gaugePopup, gaugePopupToggle } from './gauge-popup.svelte';
	import { buttonVariants } from '@ggml-org/llama-ui/ui/button';
	import * as Popover from '@ggml-org/llama-ui/ui/popover';
	import { cn } from '@ggml-org/llama-ui/utils';
	import GlassKey from '#lib/components/app/actions/GlassKey.svelte';
	import { useContextGauge } from '#lib/hooks/use-context-gauge.svelte.ts';
	import { chatStore, conversationsStore, deviceStore } from '#lib/stores/index.ts';
	import { untrack } from 'svelte';

	interface Props {
		/** Sizes and places the trigger. */
		class?: string;
		/** Draws the dial as the icon of a phone's corner key, matching the menu key's glyph. */
		corner?: boolean;
	}

	let { class: className = '', corner = false }: Props = $props();

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

<!-- a phone's tap opens the drawer the chat form renders; elsewhere a click opens the popover -->
{#if deviceStore.isMobile}
	{#if corner}
		<!-- a phone's corner key, the same glass key as the menu key across from it -->
		<GlassKey
			aria-expanded={gaugePopup.open}
			aria-haspopup="dialog"
			class={className}
			label="Context usage"
			onclick={gaugePopupToggle}
		>
			{@render dial('key')}
		</GlassKey>
	{:else}
		<button
			aria-expanded={gaugePopup.open}
			aria-haspopup="dialog"
			aria-label="Context usage"
			class={['flex size-6 items-center justify-center rounded-full', className]}
			onclick={gaugePopupToggle}
			type="button"
		>
			<!-- a phone gets the larger dial: the trigger is a touch target there -->
			{@render dial('md')}
		</button>
	{/if}
{:else}
	<Popover.Root bind:open={gaugePopup.open}>
		<Popover.Trigger
			aria-label="Context usage"
			class={cn(
				buttonVariants({ variant: 'ghost' }),
				'size-7 cursor-pointer rounded-full p-0',
				className
			)}
		>
			{@render dial('sm')}
		</Popover.Trigger>

		<!-- bits-ui sets aria-haspopup="dialog" on the trigger but gives the content no role -->
		<Popover.Content
			aria-label="Context"
			class="w-64 rounded-xl p-1.5 text-body-sm"
			role="dialog"
			side="top"
			sideOffset={8}
		>
			<ContextGaugePanel />
		</Popover.Content>
	</Popover.Root>
{/if}
