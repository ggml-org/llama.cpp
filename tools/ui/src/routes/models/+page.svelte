<script lang="ts">
	import ModelsManager from '$lib/components/app/models/ModelsManager/ModelsManager.svelte';
	import { Button } from '$lib/components/ui/button';
	import { APP_NAME, DISCOVER_ICON } from '$lib/constants';
	import { modelsStore, uiStore } from '$lib/stores';
	import { onMount } from 'svelte';

	// a direct load lands here before any chat has listed the models, and the root
	// layout fetches the router details once that list arrives. An already loaded
	// list only needs the router fetch as a sync point (a no-op outside router mode)
	onMount(() => {
		if (modelsStore.models.length > 0) {
			void modelsStore.fetchRouterModels();

			return;
		}

		modelsStore.fetch().catch((error) => console.error('Unable to load models:', error));
	});
</script>

<svelte:head>
	<title>Models - {APP_NAME}</title>
</svelte:head>

<div
	class="mx-auto flex h-[100dvh] w-full max-w-7xl flex-col gap-6 px-4 pt-[calc(3rem+var(--safe-area-top))] md:px-8 md:pt-6"
>
	<div class="flex items-start gap-4">
		<div class="flex min-w-0 flex-col gap-1">
			<h1 class="text-2xl font-semibold tracking-tight">Models</h1>

			<p class="text-sm text-muted-foreground">
				Load, configure and remove the models this server can run.
			</p>
		</div>

		{#if uiStore.isDiscoverEnabled}
			<Button
				class="ml-auto shrink-0 gap-1.5"
				onclick={() => uiStore.openModelsDiscover()}
				size="sm"
				variant="tertiary"
			>
				<DISCOVER_ICON class="h-3.5 w-3.5" />

				Discover Models
			</Button>
		{/if}
	</div>

	<ModelsManager class="min-h-0" />
</div>
