<script lang="ts">
	import { ArrowUp } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';

	interface Props {
		canSend?: boolean;
		disabled?: boolean;
		isLoading?: boolean;
		showErrorState?: boolean;
		tooltipLabel?: string;
	}

	let {
		canSend = false,
		disabled = false,
		isLoading = false,
		showErrorState = false,
		tooltipLabel
	}: Props = $props();

	let isDisabled = $derived(!canSend || disabled || isLoading);
</script>

{#snippet submitButton(props = {})}
	<Button
		type="submit"
		disabled={isDisabled}
		class={cn(
			'h-8 w-8 rounded-full p-0',
			showErrorState
				? 'bg-red-400/10 text-red-400 hover:bg-red-400/20 hover:text-red-400 disabled:opacity-100'
				: ''
		)}
		{...props}
	>
		<span class="sr-only">Send</span>
		<ArrowUp class="h-12 w-12" />
	</Button>
{/snippet}

{#if tooltipLabel}
	<Tooltip.Root>
		<Tooltip.Trigger>
			{@render submitButton()}
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>{tooltipLabel}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{:else}
	{@render submitButton()}
{/if}
