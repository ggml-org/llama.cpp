<script lang="ts">
	import BackendForm from './BackendForm.svelte';
	import BackendPresetCard from './BackendPresetCard.svelte';
	import { CheckCircle2, Loader2, PlugZap, XCircle } from '@lucide/svelte';
	import { browser } from '$app/environment';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import {
		BACKEND_ID_PREFIX,
		BACKEND_PRESETS,
		BackendProtocol,
		DISMISSED_RECOMMENDED_BACKENDS_LOCALSTORAGE_KEY
	} from '$lib/constants';
	import { BooleanString } from '$lib/enums';
	import { BackendsService } from '$lib/services';
	import type { BackendTestResult } from '$lib/services/backends.service';
	import { backendsStore } from '$lib/stores';
	import type { Backend, BackendPreset } from '$lib/types';
	import { findBackendPreset, uuid } from '$lib/utils';
	import { untrack } from 'svelte';

	interface Props {
		backend?: Backend | null;
		defaultProtocol?: BackendProtocol;
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
		onSaved?: (backend: Backend) => void;
	}

	let {
		backend = null,
		defaultProtocol = BackendProtocol.OPENAI,
		onOpenChange,
		onSaved,
		open = $bindable(false)
	}: Props = $props();

	let draft = $state<Backend>(createBackend());
	let testResult = $state<BackendTestResult | null>(null);
	let testing = $state(false);
	let detected = $state<BackendProtocol | null>(null);
	let detecting = $state(false);
	let apiKeyRequired = $state(false);
	let detectRun = 0;

	// the card follows the URL, so editing any field deselects it
	let selectedPresetId = $derived(findBackendPreset(draft.baseUrl)?.id ?? null);
	// presets a configured backend already points at
	let addedPresetIds = $derived(
		backendsStore.external
			.map((backend) => findBackendPreset(backend.baseUrl)?.id)
			.filter((id) => id !== undefined)
	);
	let unconfiguredPresets = $derived(
		BACKEND_PRESETS.filter((preset) => !addedPresetIds.includes(preset.id))
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
	let canSave = $derived(
		!urlError && draft.name.trim().length > 0 && (!apiKeyRequired || Boolean(draft.apiKey?.trim()))
	);

	// Backward-compatible read: older versions stored a JSON array of dismissed ids.
	function readRecommendationsDismissed(): boolean {
		if (!browser) return false;

		const raw = localStorage.getItem(DISMISSED_RECOMMENDED_BACKENDS_LOCALSTORAGE_KEY);

		if (!raw) return false;

		if (raw === BooleanString.TRUE) return true;

		if (raw === BooleanString.FALSE) return false;

		try {
			const parsed = JSON.parse(raw);

			return Array.isArray(parsed) && parsed.length > 0;
		} catch {
			return false;
		}
	}

	function writeRecommendationsDismissed(dismissed: boolean) {
		recommendationsDismissed = dismissed;

		if (browser) {
			localStorage.setItem(
				DISMISSED_RECOMMENDED_BACKENDS_LOCALSTORAGE_KEY,
				dismissed ? BooleanString.TRUE : BooleanString.FALSE
			);
		}
	}

	let recommendationsDismissed = $state<boolean>(readRecommendationsDismissed());

	let presetsToShow = $derived(recommendationsDismissed ? [] : unconfiguredPresets);

	// reset the draft each time the dialog opens
	$effect(() => {
		if (!open) return;

		draft = backend ? { ...backend } : createBackend();
		testResult = null;
		testing = false;
		detected = null;
		apiKeyRequired = false;
	});

	// Once the URL settles, ask the endpoint what it speaks, the way the MCP dialog
	// previews a server. A preset already knows, and the answer is applied only when
	// it differs, so this cannot feed itself.
	$effect(() => {
		const url = draft.baseUrl.trim();
		// the key is a dependency: a refused probe is retried once one is typed
		const apiKey = draft.apiKey?.trim();

		if (!open || isEdit || urlError || !url) {
			detected = null;
			detecting = false;
			apiKeyRequired = false;

			return;
		}

		const run = ++detectRun;
		const timer = setTimeout(async () => {
			detecting = true;

			const probe = await BackendsService.detectProtocol({
				...draft,
				apiKey,
				baseUrl: url
			});

			if (run !== detectRun) return;

			detecting = false;
			detected = probe.protocol;
			apiKeyRequired = probe.authRequired;

			untrack(() => {
				if (probe.protocol !== draft.protocol) handleChange({ protocol: probe.protocol });
			});
		}, 600);

		return () => clearTimeout(timer);
	});

	function createBackend(): Backend {
		return {
			baseUrl: '',
			enabled: true,
			id: uuid() || `${BACKEND_ID_PREFIX}-${Date.now()}`,
			name: '',
			protocol: defaultProtocol
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
		detected = null;
		apiKeyRequired = false;
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
			<Dialog.Title class="select-none">
				{isEdit ? 'Edit Provider' : 'Add New Provider'}
			</Dialog.Title>

			<Dialog.Description>
				Point at another llama-server, or connect an OpenAI-compatible endpoint.
			</Dialog.Description>
		</Dialog.Header>

		{#if !isEdit && presetsToShow.length > 0}
			<div class="space-y-3 pt-2">
				<div class="flex items-center justify-between gap-3">
					<h3 class="text-sm font-medium">Recommended providers</h3>

					<Button
						class="text-muted-foreground"
						onclick={() => writeRecommendationsDismissed(true)}
						size="sm"
						variant="ghost">Dismiss</Button
					>
				</div>

				<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
					{#each presetsToShow as preset (preset.id)}
						<BackendPresetCard
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
			<div class="space-y-4 py-4">
				<BackendForm
					{apiKeyRequired}
					backend={draft}
					{detected}
					{detecting}
					id="backend"
					onChange={handleChange}
					{urlError}
				/>
			</div>

			<!-- the last thing before the actions: what the endpoint said when asked -->
			<div
				aria-live="polite"
				class="mb-3 flex items-start gap-2.5 rounded-lg border border-border/60 bg-muted/30 p-3"
			>
				{#if testing}
					<Loader2 class="mt-0.5 h-4 w-4 shrink-0 animate-spin text-muted-foreground" />

					<div class="min-w-0">
						<p class="text-sm font-medium">Testing connection</p>

						<p class="text-xs text-muted-foreground">
							Asking {hostOf(draft.baseUrl)} for the models it serves.
						</p>
					</div>
				{:else if testResult?.ok}
					<CheckCircle2 class="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />

					<div class="min-w-0">
						<p class="text-sm font-medium">Connected</p>

						<p class="text-xs text-muted-foreground">
							{testResult.modelCount ?? 0}
							model{(testResult.modelCount ?? 0) === 1 ? '' : 's'} available.
						</p>
					</div>
				{:else if testResult}
					<XCircle class="mt-0.5 h-4 w-4 shrink-0 text-destructive" />

					<div class="min-w-0">
						<p class="text-sm font-medium text-destructive">Connection failed</p>

						<p class="text-xs break-words text-muted-foreground">
							{testResult.error ?? 'The endpoint did not answer.'}{testResult.status
								? ` (HTTP ${testResult.status})`
								: ''}
						</p>
					</div>
				{:else}
					<PlugZap class="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />

					<div class="min-w-0">
						<p class="text-sm font-medium">Connection not tested</p>

						<p class="text-xs text-muted-foreground">
							Test it to list the models this provider serves.
						</p>
					</div>
				{/if}
			</div>

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
