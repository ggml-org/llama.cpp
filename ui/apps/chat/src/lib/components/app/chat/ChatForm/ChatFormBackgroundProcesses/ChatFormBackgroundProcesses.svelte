<script lang="ts">
	// The chip under the chat form that counts the processes the agent left running
	// (dev servers, watchers), and the dialog it opens to inspect and stop them.
	import ChatFormBackgroundProcessRow from './ChatFormBackgroundProcessRow.svelte';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Dialog from '@ggml-org/llama-ui/ui/dialog';
	import { SquareTerminal, X } from '@lucide/svelte';
	import { backgroundProcessesStore as store } from '#lib/stores/index.ts';

	$effect(() => {
		if (!store.available) return;

		return store.watch();
	});

	const count = $derived(store.processes.length);
	const running = $derived(store.runningCount);
	const label = $derived(
		running > 0
			? `${running} background ${running === 1 ? 'process' : 'processes'}`
			: `${count} stopped ${count === 1 ? 'process' : 'processes'}`
	);
</script>

{#if store.available && count > 0}
	<button
		class="mt-1.5 flex w-auto min-w-0 items-center gap-1 rounded-md px-2 py-1 text-caption text-muted-foreground backdrop-blur-2xl hover:text-foreground"
		data-background-processes-chip
		onclick={() => (store.dialogOpen = true)}
		type="button"
	>
		<SquareTerminal class="size-icon-sm" />

		<span class="truncate">{label}</span>
	</button>
{/if}

<Dialog.Root bind:open={store.dialogOpen}>
	<Dialog.Content
		class="flex max-h-[calc(100vh-4rem)] flex-col gap-0 p-0 md:w-[calc(100vw-4rem)]! md:max-w-3xl!"
	>
		<Dialog.Header
			class="flex-row items-center gap-2 border-b border-border/40 p-4"
			showCloseButton={false}
		>
			<Dialog.Title class="text-body-sm font-semibold">Background processes</Dialog.Title>

			{#if running > 1}
				<Button
					class="h-auto px-2 py-1 text-caption"
					onclick={() => void store.stopAll()}
					size="sm"
					variant="secondary"
				>
					Stop all
				</Button>
			{/if}

			<Dialog.Close
				aria-label="Close"
				class="ml-auto inline-flex cursor-pointer items-center justify-center rounded-md p-1.5 text-muted-foreground/70 transition-colors hover:bg-muted-foreground/10 hover:text-foreground"
				type="button"
			>
				<X class="size-icon-md" />
			</Dialog.Close>
		</Dialog.Header>

		<Dialog.Description class="sr-only">
			Commands the agent started in the background, with their output
		</Dialog.Description>

		<div class="grid gap-2 overflow-y-auto p-4">
			{#each store.processes as process (process.id)}
				<ChatFormBackgroundProcessRow {process} />
			{:else}
				<p class="text-body-sm text-muted-foreground">Nothing is running in the background.</p>
			{/each}
		</div>
	</Dialog.Content>
</Dialog.Root>
