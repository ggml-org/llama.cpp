<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import { BackendCard, DialogBackendForm } from '$lib/components/app/backends';
	import { Button } from '$lib/components/ui/button';
	import { backendsModelsStore, backendsStore, serverStore } from '$lib/stores';
	import type { Backend, BackendProtocol } from '$lib/types';
	import { fade } from 'svelte/transition';

	interface Props {
		class?: string;
		/** Narrows the list to one protocol; omit it to list every provider at once. */
		protocol?: BackendProtocol;
	}

	let { class: className, protocol }: Props = $props();

	let backends = $derived(
		protocol
			? backendsStore.external.filter((b) => b.protocol === protocol)
			: backendsStore.external
	);
	// the bundled server belongs to the list whenever llama.cpp providers are in it
	let showsLocal = $derived(protocol === undefined || protocol === 'llama.cpp');
	let addTitle = $derived(
		protocol === 'openai'
			? 'Add a backend'
			: protocol === 'llama.cpp'
				? 'Add a Llama-compatible backend'
				: 'Add new provider'
	);
	let addDescription = $derived(
		protocol === 'openai'
			? 'Connect an OpenAI-compatible endpoint.'
			: protocol === 'llama.cpp'
				? 'Point at another llama-server.'
				: 'External llama-server or connect an OpenAI-compatible API.'
	);

	let isAdding = $state(false);
	let editing = $state<Backend | null>(null);

	function handleAdd() {
		editing = null;
		isAdding = true;
	}

	function handleEdit(backend: Backend) {
		editing = backend;
		isAdding = true;
	}

	function handleOpenChange(open: boolean) {
		isAdding = open;

		if (!open) {
			editing = null;
		}
	}
</script>

<div
	in:fade={{ duration: 150 }}
	class={['grid gap-4', className]}
	style="grid-template-columns: repeat(auto-fill, minmax(min(25rem, calc(100dvw - 4rem)), 1fr));"
>
	<DialogBackendForm
		bind:open={isAdding}
		backend={editing}
		defaultProtocol={protocol ?? 'llama.cpp'}
		onOpenChange={handleOpenChange}
		onSaved={() => void backendsModelsStore.loadAll()}
	/>

	{#if showsLocal && !serverStore.localServerMissing}
		<BackendCard
			backend={backendsStore.local}
			isLocal
			onToggle={(enabled) => backendsStore.setLocalEnabled(enabled)}
		/>
	{/if}

	{#each backends as backend (backend.id)}
		<BackendCard
			{backend}
			onDelete={() => {
				backendsStore.removeBackend(backend.id);
				void backendsModelsStore.loadAll();
			}}
			onEdit={() => handleEdit(backend)}
			onToggle={(enabled) => {
				backendsStore.updateBackend(backend.id, { enabled });
				void backendsModelsStore.loadAll();
			}}
		/>
	{/each}

	<div
		class="flex w-full flex-col items-center justify-center gap-4 rounded-xl border border-dashed p-6"
	>
		<Button onclick={handleAdd} size="sm">
			<Plus />

			{addTitle}
		</Button>

		<p class="text-xs text-muted-foreground">{addDescription}</p>
	</div>
</div>
