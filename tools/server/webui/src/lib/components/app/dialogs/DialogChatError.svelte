<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { AlertTriangle, TimerOff } from '@lucide/svelte';

	interface Props {
		open: boolean;
		type: 'timeout' | 'server';
		message: string;
		contextInfo?: { n_prompt_tokens: number; n_ctx: number };
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), type, message, contextInfo, onOpenChange }: Props = $props();

	const isTimeout = $derived(type === 'timeout');
	const title = $derived(isTimeout ? 'TCP Timeout' : 'Server Error');
	const description = $derived(
		isTimeout
			? 'The request did not receive a response from the server before timing out.'
			: 'The server responded with an error message. Review the details below.'
	);
	const iconClass = $derived(isTimeout ? 'text-destructive' : 'text-amber-500');
	const badgeClass = $derived(
		isTimeout
			? 'border-destructive/40 bg-destructive/10 text-destructive'
			: 'border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400'
	);

	function handleOpenChange(newOpen: boolean) {
		open = newOpen;
		onOpenChange?.(newOpen);
	}
</script>

<AlertDialog.Root {open} onOpenChange={handleOpenChange}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				{#if isTimeout}
					<TimerOff class={`h-5 w-5 ${iconClass}`} />
				{:else}
					<AlertTriangle class={`h-5 w-5 ${iconClass}`} />
				{/if}

				{title}
			</AlertDialog.Title>

			<AlertDialog.Description>
				{description}
			</AlertDialog.Description>
		</AlertDialog.Header>

		<div class={`rounded-lg border px-4 py-3 text-sm ${badgeClass}`}>
			<p class="font-medium">{message}</p>
			{#if contextInfo}
				<div class="mt-2 space-y-1 text-xs opacity-80">
					<p>
						<span class="font-medium">Prompt tokens:</span>
						{contextInfo.n_prompt_tokens.toLocaleString()}
					</p>
					<p><span class="font-medium">Context size:</span> {contextInfo.n_ctx.toLocaleString()}</p>
				</div>
			{/if}
		</div>

		<AlertDialog.Footer>
			<AlertDialog.Action onclick={() => handleOpenChange(false)}>Close</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
