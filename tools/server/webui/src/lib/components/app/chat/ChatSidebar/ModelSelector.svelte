<script lang="ts">
	import { onMount } from 'svelte';
	import { Loader2, RefreshCw } from '@lucide/svelte';
	import * as Select from '$lib/components/ui/select';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
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
	import { toast } from 'svelte-sonner';

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
			reportError('Unable to load models', error);
		} finally {
			isMounted = true;
		}
	});

	async function handleRefresh() {
		try {
			await fetchModels(true);
			toast.success('Model list refreshed');
		} catch (error) {
			reportError('Failed to refresh model list', error);
		}
	}

	async function handleSelect(value: string | undefined) {
		if (!value) return;

		const option = options.find((item) => item.id === value);
		if (!option) {
			reportError('Model is no longer available', new Error('Unknown model'));
			return;
		}

		try {
			await selectModel(option.id);
			toast.success(`Switched to ${option.name}`);
		} catch (error) {
			reportError('Failed to switch model', error);
		}
	}

	function reportError(message: string, error: unknown) {
		const description = error instanceof Error ? error.message : 'Unknown error';
		toast.error(message, { description });
	}

	function getDisplayOption(): ModelOption | undefined {
		if (activeId) {
			return options.find((option) => option.id === activeId);
		}

		return options[0];
	}

	function getCapabilityLabel(capability: string): string {
		switch (capability.toLowerCase()) {
			case 'vision':
				return 'Vision';
			case 'audio':
				return 'Audio';
			case 'multimodal':
				return 'Multimodal';
			case 'completion':
				return 'Text';
			default:
				return capability;
		}
	}
</script>

<div class="rounded-lg border border-border/40 bg-background/5 p-3 shadow-sm">
	<div class="mb-2 flex items-center justify-between">
		<p class="text-xs font-medium text-muted-foreground">Model selector</p>

		<Button
			aria-label="Refresh model list"
			class="h-7 w-7"
			disabled={loading}
			onclick={handleRefresh}
			size="icon"
			variant="ghost"
		>
			{#if loading}
				<Loader2 class="h-4 w-4 animate-spin" />
			{:else}
				<RefreshCw class="h-4 w-4" />
			{/if}
		</Button>
	</div>

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
			<Select.Trigger class="h-9 w-full justify-between">
				<span class="truncate text-sm font-medium">{selectedOption?.name || 'Select model'}</span>

				{#if updating}
					<Loader2 class="h-4 w-4 animate-spin text-muted-foreground" />
				{/if}
			</Select.Trigger>

			<Select.Content class="z-[100000]">
				{#each options as option (option.id)}
					<Select.Item value={option.id} label={option.name}>
						<div class="flex flex-col gap-1">
							<span class="text-sm font-medium">{option.name}</span>

							{#if option.description}
								<span class="text-xs text-muted-foreground">{option.description}</span>
							{/if}

							{#if option.capabilities.length > 0}
								<div class="flex flex-wrap gap-1">
									{#each option.capabilities as capability (capability)}
										<Badge variant="secondary" class="text-[10px]">
											{getCapabilityLabel(capability)}
										</Badge>
									{/each}
								</div>
							{/if}
						</div>
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>

		{#if selectedOption?.capabilities.length}
			<div class="mt-3 flex flex-wrap gap-1">
				{#each selectedOption.capabilities as capability (capability)}
					<Badge variant="outline" class="text-[10px]">
						{getCapabilityLabel(capability)}
					</Badge>
				{/each}
			</div>
		{/if}
	{/if}

	{#if error}
		<p class="mt-2 text-xs text-destructive">{error}</p>
	{/if}
</div>
