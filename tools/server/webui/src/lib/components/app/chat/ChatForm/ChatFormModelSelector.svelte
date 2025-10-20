<script lang="ts">
	import { onMount } from 'svelte';
	import { Loader2 } from '@lucide/svelte';
	import * as Select from '$lib/components/ui/select';
	import { cn } from '$lib/components/ui/utils';
	import {
		fetchModels,
		modelOptions,
		modelsError,
		modelsLoading,
		modelsUpdating,
		selectModel,
		selectedModelId
	} from '$lib/stores/models.svelte';
	import type { ModelOption } from '$lib/stores/models.svelte';

	interface Props {
		class?: string;
	}

	let { class: className = '' }: Props = $props();

	let options = $derived(modelOptions());
	let loading = $derived(modelsLoading());
	let updating = $derived(modelsUpdating());
	let error = $derived(modelsError());
	let activeId = $derived(selectedModelId());

	let isMounted = $state(false);

	onMount(async () => {
		try {
			await fetchModels();
		} catch (error) {
			console.error('Unable to load models:', error);
		} finally {
			isMounted = true;
		}
	});

	async function handleSelect(value: string | undefined) {
		if (!value) return;

		const option = options.find((item) => item.id === value);
		if (!option) {
			console.error('Model is no longer available');
			return;
		}

		try {
			await selectModel(option.id);
		} catch (error) {
			console.error('Failed to switch model:', error);
		}
	}

	function getDisplayOption(): ModelOption | undefined {
		if (activeId) {
			return options.find((option) => option.id === activeId);
		}

		return options[0];
	}
</script>

<div class={cn('flex max-w-[200px] min-w-[120px] flex-col items-end gap-1', className)}>
	{#if loading && options.length === 0 && !isMounted}
		<div class="flex items-center gap-2 text-xs text-muted-foreground">
			<Loader2 class="h-4 w-4 animate-spin" />
			Loading models…
		</div>
	{:else if options.length === 0}
		<p class="text-xs text-muted-foreground">No models available.</p>
	{:else}
		{@const selectedOption = getDisplayOption()}

		<Select.Root
			type="single"
			value={selectedOption?.id ?? ''}
			onValueChange={handleSelect}
			disabled={loading || updating}
		>
			<Select.Trigger variant="plain" size="sm" class="hover:text-foreground">
				<span class="max-w-[160px] truncate text-right"
					>{selectedOption?.name || 'Select model'}</span
				>

				{#if updating}
					<Loader2 class="h-3.5 w-3.5 animate-spin text-muted-foreground" />
				{/if}
			</Select.Trigger>

			<Select.Content class="z-[100000]">
				{#each options as option (option.id)}
					<Select.Item value={option.id} label={option.name}>
						<span class="text-sm font-medium">{option.name}</span>

						{#if option.description}
							<span class="text-xs text-muted-foreground">{option.description}</span>
						{/if}
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	{/if}

	{#if error}
		<p class="text-xs text-destructive">{error}</p>
	{/if}
</div>
