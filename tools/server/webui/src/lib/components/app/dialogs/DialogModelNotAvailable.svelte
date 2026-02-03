<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { AlertTriangle, ArrowRight } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';

	interface Props {
		open: boolean;
		modelName: string;
		availableModels?: string[];
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), modelName, availableModels = [], onOpenChange }: Props = $props();

	function handleOpenChange(newOpen: boolean) {
		open = newOpen;
		onOpenChange?.(newOpen);
	}

	function handleSelectModel(model: string) {
		// Build URL with selected model, preserving other params
		const url = new URL(page.url);
		url.searchParams.set('model', model);

		handleOpenChange(false);
		goto(url.toString());
	}
</script>

<AlertDialog.Root {open} onOpenChange={handleOpenChange}>
	<AlertDialog.Content class="max-w-lg">
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<AlertTriangle class="h-5 w-5 text-amber-500" />
				Model Not Available
			</AlertDialog.Title>

			<AlertDialog.Description>
				The requested model could not be found. Select an available model to continue.
			</AlertDialog.Description>
		</AlertDialog.Header>

		<div class="space-y-3">
			<div class="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm">
				<p class="font-medium text-amber-600 dark:text-amber-400">
					Requested: <code class="rounded bg-amber-500/20 px-1.5 py-0.5">{modelName}</code>
				</p>
			</div>

			{#if availableModels.length > 0}
				<div class="text-sm">
					<p class="mb-2 font-medium text-muted-foreground">Select an available model:</p>
					<div class="max-h-48 space-y-1 overflow-y-auto rounded-md border p-1">
						{#each availableModels as model (model)}
							<button
								type="button"
								class="group flex w-full items-center justify-between gap-2 rounded-sm px-3 py-2 text-left text-sm transition-colors hover:bg-accent hover:text-accent-foreground"
								onclick={() => handleSelectModel(model)}
							>
								<span class="min-w-0 truncate font-mono text-xs">{model}</span>
								<ArrowRight
									class="h-4 w-4 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100"
								/>
							</button>
						{/each}
					</div>
				</div>
			{/if}
		</div>

		<AlertDialog.Footer>
			<AlertDialog.Action onclick={() => handleOpenChange(false)}>Cancel</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
