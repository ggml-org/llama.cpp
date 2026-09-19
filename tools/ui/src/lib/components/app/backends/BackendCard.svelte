<script lang="ts">
	import BackendIcon from './BackendIcon.svelte';
	import { Pencil, Server, Trash2 } from '@lucide/svelte';
	import { DialogConfirmation } from '$lib/components/app/dialogs';
	import Logo from '$lib/components/app/misc/Logo.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { Switch } from '$lib/components/ui/switch';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import type { Backend, BackendProtocol } from '$lib/types';

	const PROTOCOL_LABELS: Record<BackendProtocol, string> = {
		'llama.cpp': 'llama.cpp',
		openai: 'OpenAI'
	};

	interface Props {
		backend: Backend;
		isLocal?: boolean;
		onDelete?: () => void;
		onEdit?: () => void;
		onToggle?: (enabled: boolean) => void;
	}

	let { backend, isLocal = false, onDelete, onEdit, onToggle }: Props = $props();

	let showDelete = $state(false);
	let protocolLabel = $derived(PROTOCOL_LABELS[backend.protocol]);
	let displayUrl = $derived(backend.baseUrl || 'This server');
</script>

<Card.Root class="!gap-3 bg-muted/30 p-4">
	<div class="flex items-start justify-between gap-3">
		<div class="flex min-w-0 items-center gap-2">
			<BackendIcon {backend} class={ICON_CLASS_DEFAULT}>
				{#snippet fallback()}
					{#if isLocal}
						<Logo class={ICON_CLASS_DEFAULT} style="--size: 1rem" />
					{:else}
						<Server class={ICON_CLASS_DEFAULT} />
					{/if}
				{/snippet}
			</BackendIcon>

			<div class="min-w-0">
				<p class="truncate text-sm font-medium">{backend.name}</p>

				<p class="truncate text-xs text-muted-foreground">{displayUrl}</p>
			</div>
		</div>

		<div class="flex shrink-0 items-center gap-1">
			{#if isLocal}
				<span class="rounded-md border px-2 py-0.5 text-[0.7rem] text-muted-foreground">
					Built-in
				</span>
			{/if}

			<span class="rounded-md border px-2 py-0.5 text-[0.7rem] text-muted-foreground">
				{protocolLabel}
			</span>
		</div>
	</div>

	<div class="flex items-center justify-between gap-2">
		<div class="flex items-center gap-2">
			<Switch checked={backend.enabled} onCheckedChange={(value) => onToggle?.(value)} />

			<span class="text-xs text-muted-foreground">
				{backend.enabled ? 'Enabled' : 'Disabled'}
			</span>
		</div>

		{#if !isLocal}
			<div class="flex items-center gap-1">
				<Button aria-label="Edit backend" onclick={() => onEdit?.()} size="sm" variant="ghost">
					<Pencil class="h-3.5 w-3.5" />
				</Button>

				<Button
					aria-label="Delete backend"
					onclick={() => (showDelete = true)}
					size="sm"
					variant="ghost"
				>
					<Trash2 class="h-3.5 w-3.5" />
				</Button>
			</div>
		{/if}
	</div>
</Card.Root>

<DialogConfirmation
	bind:open={showDelete}
	confirmText="Delete"
	description="This removes the backend and its stored API key."
	onCancel={() => (showDelete = false)}
	onConfirm={() => {
		showDelete = false;
		onDelete?.();
	}}
	title="Delete backend?"
	variant="destructive"
/>
