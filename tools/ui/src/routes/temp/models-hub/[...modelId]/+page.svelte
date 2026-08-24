<script lang="ts">
	import ModelsHub from '../ModelsHub.svelte';
	import { page } from '$app/state';

	let modelId = $derived(page.params.modelId ?? '');
</script>

<svelte:head>
	<title>{modelId ? `${modelId} · llama.cpp` : 'Models · llama.cpp'}</title>
</svelte:head>

<ModelsHub {modelId} />
