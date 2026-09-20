<script lang="ts">
	import BackendIcon from './BackendIcon.svelte';
	import { Pencil, Server, Trash2 } from '@lucide/svelte';
	import { DialogConfirmation } from '$lib/components/app/dialogs';
	import Logo from '$lib/components/app/misc/Logo.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { Switch } from '$lib/components/ui/switch';
	import type { Backend, BackendProtocol } from '$lib/types';

	const PROTOCOL_LABELS: Record<BackendProtocol, string> = {
		'llama.cpp': 'llama.cpp',
		openai: 'OpenAI'
	};

	const CARD_ICON_CLASS = 'h-5 w-5';

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
			<BackendIcon {backend} class={CARD_ICON_CLASS}>
				{#snippet fallback()}
					{#if isLocal}
						<Logo class={CARD_ICON_CLASS} style="--size: 1.25rem" />
					{:else}
						<Server class={CARD_ICON_CLASS} />
					{/if}
				{/snippet}
			</BackendIcon>

			<div class="min-w-0">
				<p class="truncate leading-5 font-medium">{backend.name}</p>

				<p class="truncate text-xs text-muted-foreground">{displayUrl}</p>
			</div>
		</div>

		<div class="flex shrink-0 items-center gap-1.5">
			{#if isLocal}
				<Badge class="h-5 px-1.5 text-[10px]" variant="outline">Built-in</Badge>
			{/if}

			<Badge class="h-5 px-1.5 text-[10px]" variant="outline">{protocolLabel}</Badge>
		</div>
	</div>

	<div class="flex items-center justify-between gap-4">
		<div class="flex items-center gap-2">
			<Switch checked={backend.enabled} onCheckedChange={(value) => onToggle?.(value)} />

			<span class="text-xs text-muted-foreground">
				{backend.enabled ? 'Enabled' : 'Disabled'}
			</span>
		</div>

		{#if !isLocal}
			<div class="flex shrink-0 items-center gap-1">
				<Button
					aria-label="Edit backend"
					class="relative h-7 w-7 after:absolute after:-inset-0.5 after:content-['']"
					onclick={() => onEdit?.()}
					size="icon"
					variant="ghost"
				>
					<Pencil />
				</Button>

				<Button
					aria-label="Delete backend"
					class="hover:text-destructive-foreground relative h-7 w-7 text-destructive after:absolute after:-inset-0.5 after:content-[''] hover:bg-destructive/10"
					onclick={() => (showDelete = true)}
					size="icon"
					variant="ghost"
				>
					<Trash2 />
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
