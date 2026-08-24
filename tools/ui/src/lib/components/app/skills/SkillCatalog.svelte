<script lang="ts">
	import {
		applySkillCatalogFilters,
		deriveSkillBudgetChip,
		distinctSkillProviders
	} from './skill-presentation';
	import SkillCatalogList from './SkillCatalogList.svelte';
	import SkillCatalogSearchToolbar from './SkillCatalogSearchToolbar.svelte';
	import SkillDetail from './SkillDetail.svelte';
	import SkillDiagnosticsPanel from './SkillDiagnosticsPanel.svelte';
	import { BookOpen, Circle, CircleSlash, RefreshCw } from '@lucide/svelte';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { StandalonePageShell } from '$lib/components/app';
	import { Alert, AlertDescription, AlertTitle } from '$lib/components/ui/alert';
	import { Button } from '$lib/components/ui/button';
	import * as Empty from '$lib/components/ui/empty';
	import * as Resizable from '$lib/components/ui/resizable/index.js';
	import { Skeleton } from '$lib/components/ui/skeleton';
	import {
		normalizeSkillBudget,
		ROUTES,
		SETTINGS_KEYS,
		SKILLS_PANE_SIZES_LOCALSTORAGE_KEY
	} from '$lib/constants';
	import {
		buildSkillRunSnapshot,
		resolveSkillPackOptions,
		SkillsPackingService
	} from '$lib/services';
	import { deviceStore } from '$lib/stores/device.svelte';
	import { modelsStore } from '$lib/stores/models/index.svelte';
	import { serverStore } from '$lib/stores/server.svelte';
	import { settingsStore } from '$lib/stores/settings/index.svelte';
	import { skillsStore } from '$lib/stores/skills.svelte';
	import type { SkillCatalogEntry, SkillPackedCatalog } from '$lib/types';
	import { isAbortError } from '$lib/utils';
	import { ApiError } from '$lib/utils/api-fetch';
	import { fly } from 'svelte/transition';

	interface Props {
		cwd: string | undefined;
		onRetry: () => void;
	}

	let { cwd, onRetry }: Props = $props();

	let previousRouteId = $state<string | null>(null);

	$effect(() => {
		const currentId = page.route.id;

		return () => {
			previousRouteId = currentId;
		};
	});

	function handleClose() {
		const prevIsSkills = previousRouteId === '/skills';

		if (browser && window.history.length > 1 && !prevIsSkills) {
			history.back();
		} else {
			goto(ROUTES.START);
		}
	}

	const mobile = $derived(deviceStore.isMobile);
	let selectedEntry = $state<SkillCatalogEntry | null>(null);
	const selectedId = $derived(selectedEntry?.id ?? null);

	const DEFAULT_PANE_SIZES: [number, number] = [55, 45];
	const MIN_CATALOG_PANE = 35;
	const MIN_DETAIL_PANE = 30;

	function normalizePaneSizes(value: unknown): [number, number] {
		if (
			!Array.isArray(value) ||
			value.length !== 2 ||
			typeof value[0] !== 'number' ||
			typeof value[1] !== 'number' ||
			!Number.isFinite(value[0]) ||
			!Number.isFinite(value[1]) ||
			value[0] <= 0 ||
			value[1] <= 0
		) {
			return [...DEFAULT_PANE_SIZES];
		}

		const total = value[0] + value[1];
		const catalog = Math.min(
			100 - MIN_DETAIL_PANE,
			Math.max(MIN_CATALOG_PANE, (value[0] / total) * 100)
		);

		return [catalog, 100 - catalog];
	}

	let paneSizesLoaded = false;
	let sizes = $state<[number, number]>([...DEFAULT_PANE_SIZES]);

	$effect(() => {
		if (!browser || mobile || paneSizesLoaded) return;

		paneSizesLoaded = true;
		try {
			const stored = localStorage.getItem(SKILLS_PANE_SIZES_LOCALSTORAGE_KEY);

			if (stored !== null) {
				sizes = normalizePaneSizes(JSON.parse(stored));
			}
		} catch (error) {
			console.warn(`Failed to load ${SKILLS_PANE_SIZES_LOCALSTORAGE_KEY}:`, error);
		}
	});

	// Reset dismissals and filters when the catalog reloads.
	let diagnosticsDismissed = $state(false);
	let searchQuery = $state('');
	let excludedProviders = $state<Set<string>>(new Set());
	let includeProjectSkills = $state(true);

	function handleSelect(entry: SkillCatalogEntry) {
		if (selectedId === entry.id) return;

		selectedEntry = entry;
	}

	function closeDetail() {
		selectedEntry = null;
	}

	function rememberPaneSize(index: 0 | 1, size: number) {
		// Derive the sibling so persisted sizes match the live layout.
		const next = index === 0 ? [size, 100 - size] : [100 - size, size];
		const normalized = normalizePaneSizes(next);

		sizes = normalized;

		try {
			localStorage.setItem(SKILLS_PANE_SIZES_LOCALSTORAGE_KEY, JSON.stringify(normalized));
		} catch (error) {
			console.warn(`Failed to persist ${SKILLS_PANE_SIZES_LOCALSTORAGE_KEY}:`, error);
		}
	}

	// Reset selection, dismissals, and filters when the CWD changes.
	$effect(() => {
		void cwd;
		selectedEntry = null;
		diagnosticsDismissed = false;
		searchQuery = '';
		excludedProviders = new Set();
		includeProjectSkills = true;
	});

	const budget = $derived(
		normalizeSkillBudget(settingsStore.config[SETTINGS_KEYS.MAX_SKILL_BUDGET])
	);
	const slot = $derived(skillsStore.slotFor(cwd));
	const status = $derived(slot?.status ?? 'loading');
	const error = $derived(slot?.error);
	const isUnavailable = $derived(
		status === 'error' && error instanceof ApiError && error.status === 404
	);
	const catalog = $derived(slot?.catalog);
	const isEmpty = $derived(status === 'ready' && (catalog?.skills.length ?? 0) === 0);
	const errorMessage = $derived(
		error instanceof Error ? error.message : 'The catalog could not be loaded.'
	);

	// Expand the workspace for a selected desktop entry.
	const isDesktopWorkspace = $derived(status === 'ready' && selectedEntry !== null && !mobile);

	// Pack the loaded catalog; availability changes do not affect it.
	let packed = $state<SkillPackedCatalog | null>(null);
	let packState = $state<'idle' | 'packing' | 'error'>('idle');
	let packError = $state<unknown>(null);

	const packErrorMessage = $derived(
		packError instanceof Error ? packError.message : 'The Skills catalog could not be packed.'
	);

	$effect(() => {
		void catalog;
		diagnosticsDismissed = false;
	});

	$effect(() => {
		if (status !== 'ready' || !catalog || catalog.skills.length === 0) {
			packed = null;
			packError = null;
			packState = 'idle';

			return;
		}

		const controller = new AbortController();
		const snapshot = buildSkillRunSnapshot(cwd, catalog);
		const effectiveModel = modelsStore.selectedModelName ?? modelsStore.models[0]?.model ?? '';
		const packOptions = resolveSkillPackOptions(effectiveModel, serverStore.isRouterMode, (model) =>
			modelsStore.isModelLoaded(model)
		);

		packed = null;
		packError = null;
		packState = 'packing';

		SkillsPackingService.pack(snapshot, { budget, ...packOptions, signal: controller.signal })
			.then((result) => {
				// Ignore an aborted pack that finished through estimation.
				if (controller.signal.aborted) return;

				packed = result;
				packState = 'idle';
			})
			.catch((packFailure) => {
				if (isAbortError(packFailure)) return;

				packError = packFailure;
				packState = 'error';
			});

		return () => controller.abort();
	});

	const availableProviders = $derived(distinctSkillProviders(catalog?.skills ?? []));
	const filteredEntries = $derived(
		applySkillCatalogFilters(catalog?.skills ?? [], {
			excludedProviders,
			includeProject: includeProjectSkills,
			query: searchQuery
		})
	);
	const budgetChip = $derived(packState === 'idle' ? deriveSkillBudgetChip(packed, budget) : null);

	function clearCatalogFilters() {
		searchQuery = '';
		excludedProviders = new Set();
		includeProjectSkills = true;
	}
</script>

{#snippet catalogList(open: boolean)}
	{#if catalog && catalog.skills.length > 0 && filteredEntries.length === 0}
		<div class="flex flex-1 items-center justify-center py-16">
			<Empty.Root class="max-w-md">
				<Empty.Header>
					<Empty.Media variant="icon">
						<BookOpen />
					</Empty.Media>

					<Empty.Title>No skills match</Empty.Title>

					<Empty.Description>
						No skills match your search and filters for this working directory.
					</Empty.Description>
				</Empty.Header>

				<Empty.Content>
					<Button size="sm" variant="outline" onclick={clearCatalogFilters}>Clear filters</Button>
				</Empty.Content>
			</Empty.Root>
		</div>
	{:else}
		<SkillCatalogList
			entries={filteredEntries}
			{selectedId}
			{open}
			onSelect={handleSelect}
			isDisabled={(id) => skillsStore.isDisabled(id)}
			onEnabledChange={(entry, enabled) => skillsStore.setEnabled(entry.id, enabled)}
		/>
	{/if}
{/snippet}

<StandalonePageShell
	icon={BookOpen}
	title="Skills"
	onClose={handleClose}
	class="w-full {selectedEntry ? 'h-[calc(100dvh-4rem)]' : ''}"
	headerClass="mx-auto w-full max-w-4xl"
>
	<div
		data-testid="skills-catalog-content"
		class="flex w-full flex-1 min-h-0 flex-col gap-4 p-4 md:p-8"
		class:mx-auto={!isDesktopWorkspace}
		class:max-w-4xl={!isDesktopWorkspace}
	>
		{#if status === 'loading'}
			<div aria-label="Loading skills catalog" class="flex flex-col gap-3">
				<Skeleton class="h-24 w-full" />
				<Skeleton class="h-24 w-full" />
				<Skeleton class="h-24 w-full" />
			</div>
			<p class="text-sm text-muted-foreground">Loading catalog...</p>
		{:else if status === 'error'}
			{#if isUnavailable}
				<Alert>
					<CircleSlash class="h-4 w-4" />
					<AlertTitle>Skills are not enabled</AlertTitle>
					<AlertDescription>
						The server does not expose a Skills catalog. Start it with the <code>--skills</code>
						flag to browse skills here.
					</AlertDescription>
				</Alert>
			{:else}
				<Alert variant="destructive">
					<Circle class="h-4 w-4" />
					<AlertTitle>Could not load the Skills catalog</AlertTitle>
					<AlertDescription>{errorMessage}</AlertDescription>
				</Alert>

				<div class="flex justify-start">
					<Button onclick={onRetry}>
						<RefreshCw class="h-3 w-3" />
						Retry
					</Button>
				</div>
			{/if}
		{:else if isEmpty}
			<div class="flex flex-1 items-center justify-center py-16">
				<Empty.Root class="max-w-md">
					<Empty.Header>
						<Empty.Media variant="icon">
							<BookOpen />
						</Empty.Media>

						<Empty.Title>No skills found</Empty.Title>

						<Empty.Description>
							The server returned an empty catalog for this working directory.
						</Empty.Description>
					</Empty.Header>
				</Empty.Root>
			</div>
		{:else}
			{#if !selectedEntry}
				{#if packState === 'packing'}
					<p class="text-sm text-muted-foreground">Calculating the Skills prompt budget...</p>
				{:else if packState === 'error'}
					<Alert variant="destructive">
						<Circle class="h-4 w-4" />
						<AlertTitle>Could not pack the Skills catalog</AlertTitle>
						<AlertDescription>{packErrorMessage}</AlertDescription>
					</Alert>
				{/if}

				{#if catalog}
					<SkillDiagnosticsPanel
						diagnostics={catalog.diagnostics}
						{budgetChip}
						dismissed={diagnosticsDismissed}
						onDismiss={() => (diagnosticsDismissed = true)}
					/>
				{/if}

				<SkillCatalogSearchToolbar
					providers={availableProviders}
					{excludedProviders}
					includeProject={includeProjectSkills}
					value={searchQuery}
					onQueryChange={(query) => (searchQuery = query)}
					onProvidersChange={(next) => (excludedProviders = new Set(next))}
					onIncludeProjectChange={(value) => (includeProjectSkills = value)}
				/>
			{/if}

			{#if mobile}
				{#if selectedEntry}
					<SkillDetail entry={selectedEntry} {cwd} onClose={closeDetail} mobile />
				{:else}
					{@render catalogList(selectedEntry !== null)}
				{/if}
			{:else}
				{#if selectedEntry}
					<Resizable.PaneGroup direction="horizontal" class="min-h-82">
						<Resizable.Pane
							defaultSize={sizes[0]}
							minSize={35}
							onResize={(size) => rememberPaneSize(0, size)}
						>
							<div class="h-full" in:fly|global={{ duration: 200, opacity: 1, x: 200 }}>
								{@render catalogList(true)}
							</div>
						</Resizable.Pane>

						<Resizable.Handle
							withHandle
							aria-label="Resize catalog and detail panels"
							class="data-[active=pointer]:before:shadow-[0_0_8px_hsl(var(--ring)/0.35)]"
						/>

						<Resizable.Pane
							defaultSize={sizes[1]}
							minSize={30}
							onResize={(size) => rememberPaneSize(1, size)}
						>
							<SkillDetail entry={selectedEntry} {cwd} onClose={closeDetail} mobile={false} />
						</Resizable.Pane>
					</Resizable.PaneGroup>
				{:else}
					{@render catalogList(selectedEntry !== null)}
				{/if}
			{/if}
		{/if}
	</div>
</StandalonePageShell>
