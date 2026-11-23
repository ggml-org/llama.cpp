<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import { ArrowUp, LoaderCircle } from '@lucide/svelte';

	interface Props {
		canSend?: boolean;
		disabled?: boolean;
		/** The send is on its way: it waits for something, e.g. dictated text being refined. */
		loading?: boolean;
		showErrorState?: boolean;
		tooltipLabel?: string;
	}

	let {
		canSend = false,
		disabled = false,
		loading = false,
		showErrorState = false,
		tooltipLabel
	}: Props = $props();

	let isDisabled = $derived(!canSend || disabled);
</script>

{#snippet submitButton(props = {})}
	<Button
		class={[
			'md:size-8 size-9 rounded-full p-0',
			showErrorState &&
				'[--key-edge:var(--button-danger-edge)] bg-surface-danger text-fg-danger hover:bg-surface-danger/80 hover:text-fg-danger disabled:opacity-100'
		]}
		disabled={isDisabled}
		type="submit"
		{...props}
	>
		<span class="sr-only">Send</span>

		{#if loading}
			<LoaderCircle class="size-icon-md animate-spin" />
		{:else}
			<ArrowUp class="size-icon-md" />
		{/if}
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
