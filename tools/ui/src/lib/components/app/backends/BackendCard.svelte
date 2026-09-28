<script lang="ts">
	import BackendIcon from './BackendIcon.svelte';
	import { Pencil, RotateCw, Server, Trash2 } from '@lucide/svelte';
	import { DialogConfirmation } from '$lib/components/app/dialogs';
	import Logo from '$lib/components/app/misc/Logo.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { Switch } from '$lib/components/ui/switch';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { BackendProtocol } from '$lib/constants';
	import { backendsModelsStore } from '$lib/stores/backendsModels.svelte';
	import type { Backend } from '$lib/types';

	const PROTOCOL_LABELS: Record<BackendProtocol, string> = {
		[BackendProtocol.COMPAT]: 'Llama-compatible',
		[BackendProtocol.OPENAI]: 'OpenAI-compatible'
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

	// the model listing doubles as the reachability probe
	let modelsState = $derived(backendsModelsStore.get(backend.id));
	let isChecking = $derived(backend.enabled && !modelsState.loaded && !modelsState.error);
	let isOnline = $derived(backend.enabled && modelsState.loaded && !modelsState.error);
	let isOffline = $derived(backend.enabled && !!modelsState.error);
	let hasStatus = $derived(isChecking || isOnline || isOffline);
	let modelCount = $derived(modelsState.models.length);
	let metaText = $derived(isLocal ? `Built-in · ${protocolLabel}` : protocolLabel);

	// kick off the probe for backends the prefetch has not covered yet
	$effect(() => {
		if (isChecking && !modelsState.loading) {
			void backendsModelsStore.ensureLoaded(backend.id);
		}
	});

	function handleRetry() {
		backendsModelsStore.clear(backend.id);
		void backendsModelsStore.ensureLoaded(backend.id);
	}
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

		<Switch
			aria-label={backend.enabled ? 'Disable backend' : 'Enable backend'}
			checked={backend.enabled}
			onCheckedChange={(value) => onToggle?.(value)}
		/>
	</div>

	{#if isOffline && modelsState.error}
		<Tooltip.Root>
			<Tooltip.Trigger>
				<p class="max-w-full truncate text-left text-xs text-destructive">{modelsState.error}</p>
			</Tooltip.Trigger>

			<Tooltip.Content>
				<p class="max-w-80 break-words">{modelsState.error}</p>
			</Tooltip.Content>
		</Tooltip.Root>
	{/if}

	<div class="mt-auto flex items-center justify-between gap-4">
		<div class="flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
			{#if isChecking}
				<span class="size-1.5 shrink-0 animate-pulse rounded-full bg-amber-500"></span>

				<span class="shrink-0">Checking</span>
			{:else if isOnline}
				<span class="size-1.5 shrink-0 rounded-full bg-emerald-500"></span>

				<span class="shrink-0">{modelCount === 1 ? '1 model' : `${modelCount} models`}</span>
			{:else if isOffline}
				<span class="size-1.5 shrink-0 rounded-full bg-destructive"></span>

				<span class="shrink-0">Unreachable</span>

				<button
					aria-label="Retry connection"
					class="shrink-0 rounded-sm p-0.5 transition-colors duration-150 ease-out hover:bg-muted hover:text-foreground"
					onclick={handleRetry}
				>
					<RotateCw class="h-3 w-3" />
				</button>
			{/if}

			{#if hasStatus}
				<span class="shrink-0 opacity-50">·</span>
			{/if}

			<span class="truncate">{metaText}</span>
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
