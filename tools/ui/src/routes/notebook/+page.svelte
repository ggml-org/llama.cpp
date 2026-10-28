<script lang="ts">
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { NotebookScreen } from '$lib/components/app';
	import { ROUTES } from '$lib/constants';
	import { settingsStore, uiStore } from '$lib/stores';

	// The notebook is opt-in from the General settings: without it, bounce back to a
	// new-chat tab without a history entry
	$effect(() => {
		if (browser && settingsStore.isInitialized && !uiStore.isNotebookEnabled) {
			goto(ROUTES.START, { replaceState: true });
		}
	});
</script>

<svelte:head>
	<title>Notebook - llama.cpp</title>
</svelte:head>

{#if uiStore.isNotebookEnabled}
	<NotebookScreen />
{/if}
