<!--
	A phone's model picker: a sheet in the settings sheet's patterns. It opens partway with the
	few models in reach (loaded, favorite, recent), each marked when it is the chat's or the
	dictation's, and a row that opens all models as a page, in groups of the
	same rows. Tapping a model switches the chat to it, loading it if need be, as on a desktop;
	holding it opens its menu, as a held chat row does: unload it, use it for dictation when it
	hears, or open its details over the sheet. A search shows its results in place; it stays at
	the sheet's foot, under the reasoning effort as the desktop picker has it.
-->
<script lang="ts">
	import ModelAvatar from '../ModelAvatar.svelte';
	import ModelId from '../ModelId.svelte';
	import ModelVoiceRole from '../ModelVoiceRole.svelte';
	import ModelsSelectorReasoning from './ModelsSelectorReasoning.svelte';
	import { SearchInput } from '@ggml-org/llama-ui/app/forms';
	import {
		InsetList,
		InsetListRow,
		SheetHeader,
		SheetPages,
		useSheetPages
	} from '@ggml-org/llama-ui/app/sheets';
	import { SHEET_CONTENT_CLASS } from '@ggml-org/llama-ui/constants';
	import * as Drawer from '@ggml-org/llama-ui/ui/drawer';
	import { Check, Info, Loader2, Mic, MicOff, MoreHorizontal, PowerOff } from '@lucide/svelte';
	import DropdownMenuActions from '#lib/components/app/navigation/DropdownMenuActions.svelte';
	import {
		MODELS_SHEET_GROUP_KEYS,
		MODELS_SHEET_TEXT,
		ROW_LONG_PRESS_MS,
		SETTINGS_KEYS,
		SHEET_AUTO_HEIGHT_CLASS,
		SHEET_FULL_HEIGHT_CLASS,
		VOICE_DICTATION_MODEL_LABEL
	} from '#lib/constants/index.ts';
	import { ModelsSheetPage } from '#lib/enums/index.ts';
	import { useLongPress } from '#lib/hooks/use-long-press.svelte.ts';
	import { useReasoningMenu } from '#lib/hooks/use-reasoning-menu.svelte.ts';
	import {
		modelsStore,
		serverStore,
		settingsStore,
		uiStore,
		voiceStore
	} from '#lib/stores/index.ts';
	import type { MenuAction } from '#lib/types/index.ts';
	import {
		type GroupedModelOptions,
		groupModelFamilies,
		type ModelItem
	} from '#lib/utils/index.ts';
	import { tick } from 'svelte';

	interface Props {
		/** The chat's model, marked among the rows. */
		activeId?: string | null;
		emptyMessage?: string;
		/** Favorite models, after the loaded ones on the All models page. */
		favoriteItems?: ModelItem[];
		/** The other models: grouped by family on the All models page, as the setting asks. */
		groups?: GroupedModelOptions;
		/** Hide the organization in the rows' model ids. */
		hideOrgName?: boolean;
		/** The row the search's arrow keys are on, in the full list. */
		highlightedId?: string | null;
		isEmpty?: boolean;
		/** Loaded models, leading the All models page. */
		loadedItems?: ModelItem[];
		onOpenChange?: (open: boolean) => void;
		onSearchChange?: (value: string) => void;
		onSearchKeyDown?: (event: KeyboardEvent) => void;
		/** A tapped model: switches the chat to it, loading it if need be. */
		onSelect?: (modelId: string) => void;
		open?: boolean;
		/** The few models the sheet opens with: the chat's, the dictation's, and those in reach. */
		quickItems?: ModelItem[];
		searchTerm?: string;
		/** The chat's reasoning effort, as a row that opens its levels. */
		showReasoning?: boolean;
		/** The chat's model when the server no longer lists it, shown as not available. */
		unavailableModel?: string | null;
	}

	let {
		activeId = null,
		emptyMessage = MODELS_SHEET_TEXT.EMPTY,
		favoriteItems = [],
		groups = { available: [] },
		hideOrgName = false,
		highlightedId = null,
		isEmpty = false,
		loadedItems = [],
		onOpenChange,
		onSearchChange,
		onSearchKeyDown,
		onSelect,
		open = $bindable(false),
		quickItems = [],
		searchTerm = '',
		showReasoning = false,
		unavailableModel = null
	}: Props = $props();

	/** Marks a model's row, to bring the highlighted one into view. */
	const MODEL_ID_ATTR = 'data-model-id';

	const pages = useSheetPages({ id: ModelsSheetPage.ROOT });
	const page = $derived(pages.page);

	const reasoning = useReasoningMenu();

	let scrollEl = $state<HTMLDivElement | null>(null);

	let isSearching = $derived(searchTerm.trim().length > 0);
	// the full list: its own page or a search's results; the sheet opens on the short view even
	// when no model is in reach, with the way to all of them
	let showFull = $derived(page.id === ModelsSheetPage.ALL || isSearching);
	let hasEffort = $derived(showReasoning && reasoning.levels.length > 0);

	// the models past the loaded and favorite ones: a group per family, or one for them all
	let otherGroups = $derived.by(() => {
		const items = groups.available.flatMap((group) => group.items);

		if (items.length === 0) return [];

		if (!settingsStore.config[SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY]) {
			return [{ items, key: MODELS_SHEET_GROUP_KEYS.LOCAL, title: MODELS_SHEET_TEXT.LOCAL_MODELS }];
		}

		return groupModelFamilies(items, (item) => item.option.model).map((family) => ({
			items: family.entries,
			key: `${MODELS_SHEET_GROUP_KEYS.FAMILY_PREFIX}${family.key}`,
			title: family.label
		}));
	});
	let fullGroups = $derived([
		...(loadedItems.length > 0
			? [
					{
						items: loadedItems,
						key: MODELS_SHEET_GROUP_KEYS.LOADED,
						title: MODELS_SHEET_TEXT.LOADED
					}
				]
			: []),
		...(favoriteItems.length > 0
			? [
					{
						items: favoriteItems,
						key: MODELS_SHEET_GROUP_KEYS.FAVORITES,
						title: MODELS_SHEET_TEXT.FAVORITES
					}
				]
			: []),
		...otherGroups
	]);

	// a held row's menu, anchored to the row; raw, so a row's item still equals the one held
	let menuItem = $state.raw<ModelItem | null>(null);
	let menuAnchor = $state<HTMLElement | null>(null);
	let menuOpen = $state(false);
	// the row under the finger, for the hold to open its menu
	let pressed = $state.raw<{ item: ModelItem; row: HTMLElement } | null>(null);

	const longPress = useLongPress({
		enabled: () => true,
		onLongPress: () => {
			if (!pressed) return;

			menuItem = pressed.item;
			menuAnchor = pressed.row;
			menuOpen = true;
		}
	});

	let menuActions = $derived.by((): MenuAction[] => {
		const item = menuItem;

		if (!item) return [];

		const { id, model } = item.option;
		const actions: MenuAction[] = [];

		// a router unloads its models; a lone server keeps its one model loaded
		if (serverStore.isRouterMode) {
			if (modelsStore.isModelLoading(model)) {
				actions.push({
					icon: Loader2,
					label: MODELS_SHEET_TEXT.CANCEL_LOADING,
					onclick: () => void modelsStore.status.cancelLoad(model)
				});
			} else if (modelsStore.isModelLoaded(model)) {
				actions.push({
					icon: PowerOff,
					label: MODELS_SHEET_TEXT.UNLOAD,
					onclick: () => void modelsStore.status.unload(model)
				});
			}
		}

		// as the desktop row's microphone: a router model that hears can be chosen to transcribe
		// dictation (the one Automatic picked is kept so), and the chosen one can hand the choice
		// back to Automatic
		if (serverStore.isRouterMode && voiceStore.roleOf(model) !== null) {
			const dictates = voiceStore.isDictationModel(model);
			const chosen = dictates && !voiceStore.dictationIsAutomatic;

			let label: string = VOICE_DICTATION_MODEL_LABEL.use;

			if (chosen) label = VOICE_DICTATION_MODEL_LABEL.pickAutomatically;
			else if (dictates) label = VOICE_DICTATION_MODEL_LABEL.keep;

			actions.push({
				icon: chosen ? MicOff : Mic,
				label,
				onclick: () => voiceStore.toggleDictation(model)
			});
		}

		actions.push({
			icon: Info,
			label: MODELS_SHEET_TEXT.MORE_DETAILS,
			onclick: () => uiStore.openModelDetails(id)
		});

		return actions;
	});

	// each opening starts on the first page
	$effect(() => {
		if (open) pages.reset();
	});

	// the arrow keys can move the highlight to a row out of view
	$effect(() => {
		if (!highlightedId || !showFull) return;

		scrollEl
			?.querySelector(`[${MODEL_ID_ATTR}="${CSS.escape(highlightedId)}"]`)
			?.scrollIntoView({ block: 'nearest' });
	});

	function openAllModels() {
		pages.push({ id: ModelsSheetPage.ALL });
		void tick().then(() => scrollEl?.scrollTo({ top: 0 }));
	}

	// the keys move a highlight through the full list, so they wait until it shows
	function handleSearchKeyDown(event: KeyboardEvent) {
		if (showFull) onSearchKeyDown?.(event);
	}
</script>

{#snippet modelRow(item: ModelItem)}
	{@const inChat = activeId === item.option.id}
	{@const dictates = !inChat && voiceStore.isDictationModel(item.option.model)}
	{@const isHeld = longPress.isPressing && pressed?.item === item}
	{@const isLit = (menuOpen && menuItem === item) || (showFull && highlightedId === item.option.id)}

	<!-- the row sinks while a finger holds it, and stays lit while its menu is open or the
	     search's arrow keys are on it -->
	<InsetListRow
		bodyClass="gap-3"
		class={[
			'select-none [-webkit-touch-callout:none] [content-visibility:auto] [contain-intrinsic-size:auto_3rem]',
			isHeld && 'bg-(--surface-pressed) duration-(--row-hold-ms) ease-out',
			isLit && 'bg-(--surface-pressed)'
		]}
		style="--row-hold-ms: {ROW_LONG_PRESS_MS}ms"
		{...{ [MODEL_ID_ATTR]: item.option.id }}
		onclick={() => onSelect?.(item.option.id)}
		onclickcapture={longPress.handleClickCapture}
		oncontextmenu={longPress.handleContextMenu}
		onpointercancel={longPress.handlePointerEnd}
		onpointerdown={(event) => {
			pressed = { item, row: event.currentTarget };
			longPress.handlePointerDown(event);
		}}
		onpointermove={longPress.handlePointerMove}
		onpointerup={longPress.handlePointerEnd}
	>
		{#snippet leading()}
			<ModelAvatar class="shrink-0" option={item.option} size="size-6" />
		{/snippet}

		<span class="flex min-w-0 flex-1 items-center gap-1.5">
			<ModelId
				class="min-w-0"
				draftSidecars={item.option.draftSidecars}
				hideCapabilities
				hideModalities
				{hideOrgName}
				modelId={item.option.model}
				tags={item.option.tags}
			/>

			<ModelVoiceRole model={item.option.model} />
		</span>

		<!-- what the model does now: the chat's, or the one that transcribes dictation -->
		{#if inChat}
			{@render role(MODELS_SHEET_TEXT.IN_CHAT, Check, 'text-(--action-primary)')}
		{:else if dictates}
			{@render role(MODELS_SHEET_TEXT.DICTATION, Mic, 'text-destructive')}
		{/if}
	</InsetListRow>
{/snippet}

{#snippet role(label: string, Icon: typeof Check, iconClass: string)}
	<span class="flex shrink-0 items-center gap-1 text-caption text-muted-foreground">
		{label}

		<Icon class="size-4 {iconClass}" />
	</span>
{/snippet}

<!-- the chat's model, when the server no longer lists it: shown, and not to be picked -->
{#snippet unavailableRow(model: string)}
	<InsetList>
		<InsetListRow aria-disabled="true" as="div" bodyClass="gap-3 text-fg-danger">
			<ModelId class="min-w-0 flex-1" {hideOrgName} hideQuantization modelId={model} />

			<span class="shrink-0 text-caption">{MODELS_SHEET_TEXT.NOT_AVAILABLE}</span>
		</InsetListRow>
	</InsetList>
{/snippet}

<!-- a held row's menu: it shows no trigger, anchored to the row as a held chat row's is -->
<div hidden>
	<DropdownMenuActions
		bind:open={menuOpen}
		actions={menuActions}
		align="center"
		anchor={menuAnchor}
		triggerIcon={MoreHorizontal}
		triggerTooltip={MODELS_SHEET_TEXT.MODEL_ACTIONS}
	/>
</div>

<!-- the details drawer opens over the picker: the picker stays open under its backdrop -->
<Drawer.Root bind:open {onOpenChange} shouldScaleBackground={false}>
	<Drawer.Content
		class={[
			'gap-0 overflow-hidden',
			showFull ? SHEET_FULL_HEIGHT_CLASS : SHEET_AUTO_HEIGHT_CLASS,
			SHEET_CONTENT_CLASS,
			uiStore.isModelDetailsOpen && 'z-40'
		]}
		onOpenAutoFocus={(event) => event.preventDefault()}
		overlayClass={uiStore.isModelDetailsOpen ? 'z-40' : undefined}
	>
		<SheetHeader
			canGoBack={pages.canGoBack}
			onBack={pages.pop}
			title={page.id === ModelsSheetPage.ALL
				? MODELS_SHEET_TEXT.ALL_MODELS
				: MODELS_SHEET_TEXT.SELECT_MODEL}
		/>

		<Drawer.Description class="sr-only">{MODELS_SHEET_TEXT.DESCRIPTION}</Drawer.Description>

		<SheetPages bind:ref={scrollEl} class="gap-6 px-4 pb-4" pageKey={page.id} {pages}>
			{#if unavailableModel}
				{@render unavailableRow(unavailableModel)}
			{/if}

			{#if showFull}
				{#if isEmpty}
					<p class="px-4 py-3 text-body-sm text-muted-foreground">{emptyMessage}</p>
				{:else}
					{#each fullGroups as group (group.key)}
						<InsetList title={group.title}>
							{#each group.items as item (item.option.id)}
								{@render modelRow(item)}
							{/each}
						</InsetList>
					{/each}
				{/if}
			{:else}
				<!-- plain rows in one rounded group, as the add sheet's -->
				{#if quickItems.length > 0}
					<InsetList>
						{#each quickItems as item (item.option.id)}
							{@render modelRow(item)}
						{/each}
					</InsetList>
				{:else if isEmpty}
					<p class="px-4 text-body-sm text-muted-foreground">{emptyMessage}</p>
				{/if}

				<InsetList>
					<InsetListRow chevron onclick={openAllModels}>
						<span class="flex-1">{MODELS_SHEET_TEXT.ALL_MODELS}</span>
					</InsetListRow>
				</InsetList>
			{/if}
		</SheetPages>

		<!-- the reasoning effort, then the search at the sheet's foot: a plain text field in the chat
		     form's shape; both keep the lists' gutter -->
		<div class="flex shrink-0 flex-col gap-3 px-4 pt-2 pb-[max(0.5rem,var(--safe-area-bottom))]">
			<!-- a search gives the sheet to its results: the reasoning row steps aside until it is
			     cleared, and stays mounted so its levels are as they were left -->
			{#if hasEffort}
				<div class={isSearching ? 'hidden' : 'contents'}>
					<ModelsSelectorReasoning variant="list" />
				</div>
			{/if}

			<SearchInput
				inputClass="h-12 max-md:h-12 rounded-(--chat-form-single-row-radius)"
				onInput={(value) => onSearchChange?.(value)}
				onKeyDown={handleSearchKeyDown}
				placeholder={MODELS_SHEET_TEXT.SEARCH_PLACEHOLDER}
				value={searchTerm}
			/>
		</div>
	</Drawer.Content>
</Drawer.Root>
