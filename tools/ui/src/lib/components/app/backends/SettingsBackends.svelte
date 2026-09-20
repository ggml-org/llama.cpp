<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import { BackendCard, DialogBackendForm } from '$lib/components/app/backends';
	import { Button } from '$lib/components/ui/button';
	import * as Empty from '$lib/components/ui/empty';
	import { backendsModelsStore, backendsStore, serverStore } from '$lib/stores';
	import type { Backend, BackendProtocol } from '$lib/types';
	import { fade } from 'svelte/transition';

	interface Props {
		class?: string;
		protocol: BackendProtocol;
	}

	let { class: className, protocol }: Props = $props();

	// the switcher above the list narrows it to one protocol
	let backends = $derived(backendsStore.external.filter((b) => b.protocol === protocol));
	let isLlamaCpp = $derived(protocol === 'llama.cpp');

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

<div in:fade={{ duration: 150 }} class={['grid gap-4', className]}>
	<DialogBackendForm
		bind:open={isAdding}
		backend={editing}
		defaultProtocol={protocol}
		onOpenChange={handleOpenChange}
		onSaved={() => void backendsModelsStore.loadAll()}
	/>

	{#if isLlamaCpp && !serverStore.localServerMissing}
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

	<Empty.Root class="border">
		<Empty.Header>
			<Empty.Media variant="icon">
				<Plus />
			</Empty.Media>

			<Empty.Title>{isLlamaCpp ? 'Add a llama.cpp backend' : 'Add a backend'}</Empty.Title>

			<Empty.Description>
				{isLlamaCpp ? 'Point at another llama-server.' : 'Connect an OpenAI-compatible endpoint.'}
			</Empty.Description>
		</Empty.Header>

		<Empty.Content>
			<Button onclick={handleAdd} size="sm">
				<Plus />

				Add Backend
			</Button>
		</Empty.Content>
	</Empty.Root>
</div>
