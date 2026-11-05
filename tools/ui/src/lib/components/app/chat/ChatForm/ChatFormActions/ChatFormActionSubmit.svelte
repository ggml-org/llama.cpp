<script lang="ts">
	import { ArrowUp } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		canSend?: boolean;
		disabled?: boolean;
		showErrorState?: boolean;
		tooltipLabel?: string;
	}

	let { canSend = false, disabled = false, showErrorState = false, tooltipLabel }: Props = $props();

	let isDisabled = $derived(!canSend || disabled);
</script>

{#snippet submitButton(props = {})}
	<Button
		class={[
			'md:size-8 size-9 rounded-full p-0',
			showErrorState &&
				'bg-surface-danger text-fg-danger hover:bg-surface-danger/80 hover:text-fg-danger disabled:opacity-100'
		]}
		disabled={isDisabled}
		type="submit"
		{...props}
	>
		<span class="sr-only">Send</span>

		<ArrowUp class="size-icon-md" />
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
