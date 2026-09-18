<script lang="ts">
	import BackendForm from './BackendForm.svelte';
	import BackendPresetCard from './BackendPresetCard.svelte';
	import { CheckCircle2, Loader2, XCircle } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { BACKEND_ID_PREFIX, BACKEND_PRESETS } from '$lib/constants';
	import { BackendsService } from '$lib/services';
	import type { BackendTestResult } from '$lib/services/backends.service';
	import { backendsStore } from '$lib/stores';
	import type { Backend, BackendPreset } from '$lib/types';
	import { findBackendPreset, uuid } from '$lib/utils';

	interface Props {
		backend?: Backend | null;
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
		onSaved?: (backend: Backend) => void;
	}

	let { backend = null, onOpenChange, onSaved, open = $bindable(false) }: Props = $props();

	let draft = $state<Backend>(createBackend());
	let testResult = $state<BackendTestResult | null>(null);
	let testing = $state(false);

	// the card follows the URL, so editing any field deselects it
	let selectedPresetId = $derived(findBackendPreset(draft.baseUrl)?.id ?? null);
	// presets a configured backend already points at
	let addedPresetIds = $derived(
		backendsStore.external
			.map((backend) => findBackendPreset(backend.baseUrl)?.id)
			.filter((id) => id !== undefined)
	);

	let isEdit = $derived(backend !== null);
	let urlError = $derived.by(() => {
		const url = draft.baseUrl.trim();

		if (!url) return 'Base URL is required';

		try {
			new URL(url);

			return null;
		} catch {
			return 'Invalid URL format';
		}
	});
	let canSave = $derived(!urlError && draft.name.trim().length > 0);

	// reset the draft each time the dialog opens
	$effect(() => {
		if (!open) return;

		draft = backend ? { ...backend } : createBackend();
		testResult = null;
		testing = false;
	});

	function createBackend(): Backend {
		return {
			baseUrl: '',
			enabled: true,
			id: uuid() || `${BACKEND_ID_PREFIX}-${Date.now()}`,
			name: '',
			protocol: 'openai'
		};
	}

	function applyPreset(preset: BackendPreset) {
		draft = {
			...draft,
			baseUrl: preset.baseUrl,
			chatPath: preset.chatPath,
			compat: preset.compat,
			modelsPath: preset.modelsPath,
			name: preset.name,
			protocol: preset.protocol
		};
		testResult = null;
	}

	function handleChange(patch: Partial<Backend>) {
		draft = { ...draft, ...patch };
		testResult = null;
	}

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}

	async function handleTest() {
		if (urlError) return;

		testing = true;
		testResult = null;

		try {
			testResult = await BackendsService.test(draft);
		} finally {
			testing = false;
		}
	}

	function handleSave() {
		if (!canSave) return;

		const next: Backend = { ...draft, name: draft.name.trim() || hostOf(draft.baseUrl) };

		if (isEdit && backend) {
			backendsStore.updateBackend(backend.id, next);
		} else {
			backendsStore.addBackend(next);
		}

		onSaved?.(next);
		handleOpenChange(false);
	}

	function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		handleSave();
	}

	function hostOf(url: string): string {
		try {
			return new URL(url).host;
		} catch {
			return url;
		}
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content class="max-w-2xl!">
		<Dialog.Header>
			<Dialog.Title>{isEdit ? 'Edit backend' : 'Add backend'}</Dialog.Title>

			<Dialog.Description>Connect an OpenAI-compatible endpoint.</Dialog.Description>
		</Dialog.Header>

		{#if !isEdit}
			<div class="space-y-3 pt-2">
				<h3 class="text-sm font-medium">Recommended providers</h3>

				<!-- TODO: a "pair by QR code" entry point belongs in this grid, next
				     to the provider cards. -->

				<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
					{#each BACKEND_PRESETS as preset (preset.id)}
						<BackendPresetCard
							added={addedPresetIds.includes(preset.id)}
							dimmed={Boolean(selectedPresetId) && selectedPresetId !== preset.id}
							onClick={() => applyPreset(preset)}
							{preset}
							selected={selectedPresetId === preset.id}
						/>
					{/each}
				</div>
			</div>
		{/if}

		<form class="contents" onsubmit={handleSubmit}>
			<div class="py-4">
				<BackendForm backend={draft} id="backend" onChange={handleChange} {urlError} />
			</div>

			{#if testing || testResult}
				<div class="flex items-center gap-2 pb-2 text-xs">
					{#if testing}
						<Loader2 class="h-3.5 w-3.5 shrink-0 animate-spin" />

						<span class="text-muted-foreground">Testing connection...</span>
					{:else if testResult?.ok}
						<CheckCircle2 class="h-3.5 w-3.5 shrink-0 text-emerald-500" />

						<span class="text-muted-foreground">
							Connected.
							{testResult.modelCount ?? 0}
							model{(testResult.modelCount ?? 0) === 1 ? '' : 's'} available.
						</span>
					{:else}
						<XCircle class="h-3.5 w-3.5 shrink-0 text-destructive" />

						<span class="text-destructive">{testResult?.error ?? 'Connection failed'}</span>
					{/if}
				</div>
			{/if}

			<Dialog.Footer>
				<Button onclick={() => handleOpenChange(false)} size="sm" variant="secondary">
					Cancel
				</Button>

				<Button disabled={testing} onclick={handleTest} size="sm" type="button" variant="outline">
					Test connection
				</Button>

				<Button disabled={!canSave} size="sm" type="submit">
					{isEdit ? 'Save' : 'Add'}
				</Button>
			</Dialog.Footer>
		</form>
	</Dialog.Content>
</Dialog.Root>
