<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import { BackendCard, DialogBackendForm } from '$lib/components/app/backends';
	import { Button } from '$lib/components/ui/button';
	import * as Empty from '$lib/components/ui/empty';
	import { backendsStore } from '$lib/stores';
	import type { Backend } from '$lib/types';
	import { fade } from 'svelte/transition';

	interface Props {
		class?: string;
	}

	let { class: className }: Props = $props();

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

<div in:fade={{ duration: 150 }} class={['flex flex-col gap-4', className]}>
	<DialogBackendForm bind:open={isAdding} backend={editing} onOpenChange={handleOpenChange} />

	<BackendCard backend={backendsStore.local} isLocal />

	{#each backendsStore.external as backend (backend.id)}
		<BackendCard
			{backend}
			onDelete={() => backendsStore.removeBackend(backend.id)}
			onEdit={() => handleEdit(backend)}
			onToggle={(enabled) => backendsStore.updateBackend(backend.id, { enabled })}
		/>
	{/each}

	<Empty.Root class="border">
		<Empty.Header>
			<Empty.Media variant="icon">
				<Plus />
			</Empty.Media>

			<Empty.Title>Add another backend</Empty.Title>

			<Empty.Description>Connect an OpenAI- or Anthropic-compatible endpoint.</Empty.Description>
		</Empty.Header>

		<Empty.Content>
			<Button onclick={handleAdd} size="sm">
				<Plus />

				Add Backend
			</Button>
		</Empty.Content>
	</Empty.Root>
</div>
