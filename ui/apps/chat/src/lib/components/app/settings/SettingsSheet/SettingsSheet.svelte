<!--
	A phone's settings: a full-height sheet. Its first page has the theme and a row per section; a row
	pushes the section's page, and the Tools row the tools. A change applies as it is made, the native
	way, so there is nothing to save.
-->
<script lang="ts">
	import SettingsSheetTools, { type SettingsToolsView } from './SettingsSheetTools.svelte';
	import {
		InsetList,
		InsetListRow,
		SheetHeader,
		SheetPages,
		useSheetPages
	} from '@ggml-org/llama-ui/app/sheets';
	import { SHEET_CONTENT_CLASS } from '@ggml-org/llama-ui/constants';
	import * as AlertDialog from '@ggml-org/llama-ui/ui/alert-dialog';
	import * as Drawer from '@ggml-org/llama-ui/ui/drawer';
	import { RefreshCw, RotateCcw } from '@lucide/svelte';
	import {
		SettingsChatFieldsInset,
		SettingsChatImportExportTab,
		SettingsChatVoiceTab,
		SettingsMcpServers
	} from '#lib/components/app/settings/index.ts';
	import {
		NUMERIC_FIELDS,
		POSITIVE_INTEGER_FIELDS,
		SETTINGS_CHAT_SECTIONS,
		SETTINGS_KEYS,
		SETTINGS_SECTION_SLUGS,
		SETTINGS_SHEET_TEXT,
		SETTINGS_SHEET_THEMES,
		settingsChatSections,
		SHEET_FULL_HEIGHT_CLASS
	} from '#lib/constants/index.ts';
	import { SettingsSheetPage } from '#lib/enums/index.ts';
	import { ColorMode } from '#lib/enums/ui.enums.ts';
	import { useToolsPanel } from '#lib/hooks/use-tools-panel.svelte.ts';
	import { agenticMode } from '#lib/stores/agentic-mode.svelte.ts';
	import {
		deviceStore,
		modelsStore,
		serverStore,
		settingsStore,
		toolsStore
	} from '#lib/stores/index.ts';
	import { getToolDisplay, isSettingShownOn } from '#lib/utils/index.ts';
	import { setMode } from 'mode-watcher';
	import { tick } from 'svelte';
	import { toast } from 'svelte-sonner';

	interface Props {
		open?: boolean;
	}

	let { open = $bindable(false) }: Props = $props();

	type Page =
		| { id: SettingsSheetPage.ROOT }
		| { id: SettingsSheetPage.SECTION; slug: string }
		| { id: SettingsSheetPage.MCP_SERVERS }
		| SettingsToolsView;

	/** The smallest value a positive whole number setting takes when its field names none. */
	const POSITIVE_INTEGER_MIN = 1;

	const pages = useSheetPages<Page>({ id: SettingsSheetPage.ROOT });
	const page = $derived(pages.page);

	const toolsPanel = useToolsPanel();

	// a plain chat has no Agent section, as in the settings dialog
	const sections = $derived(settingsChatSections(agenticMode.enabled));
	const allFields = SETTINGS_CHAT_SECTIONS.flatMap((entry) => entry.fields ?? []);

	let config = $derived(settingsStore.config);
	let showResetDialog = $state(false);

	let section = $derived(
		page.id === SettingsSheetPage.SECTION
			? SETTINGS_CHAT_SECTIONS.find((entry) => entry.slug === page.slug)
			: undefined
	);

	// the theme has the first page's tiles, so a section does not list it again
	let sectionFields = $derived(
		(section?.fields ?? []).filter(
			(field) => field.key !== SETTINGS_KEYS.THEME && isSettingShownOn(field, deviceStore.isMobile)
		)
	);

	const title = $derived.by(() => {
		switch (page.id) {
			case SettingsSheetPage.SECTION:
				return section?.title ?? SETTINGS_SHEET_TEXT.TITLE;
			case SettingsSheetPage.MCP_SERVERS:
				return SETTINGS_SHEET_TEXT.MCP_SERVERS;
			case SettingsSheetPage.TOOLS:
				return SETTINGS_SHEET_TEXT.TOOLS;
			case SettingsSheetPage.TOOL_GROUP:
				return (
					toolsStore.toolGroups.find((group) => group.key === page.groupKey)?.label ??
					SETTINGS_SHEET_TEXT.TOOLS
				);
			case SettingsSheetPage.TOOL: {
				const entry = toolsStore.toolGroups
					.flatMap((group) => group.tools)
					.find((tool) => tool.key === page.toolKey);

				return entry ? getToolDisplay(entry).label : SETTINGS_SHEET_TEXT.TOOL;
			}
			default:
				return SETTINGS_SHEET_TEXT.TITLE;
		}
	});

	// a section's number fields show the server's defaults, which a router reads off its models
	let fetchInitiated = false;

	$effect(() => {
		if (open && serverStore.isRouterMode && section?.fields?.length && !fetchInitiated) {
			fetchInitiated = true;

			void modelsStore
				.fetch()
				.then(() => modelsStore.fetchRouterModels())
				.then(() => modelsStore.props.fetchModalitiesForLoadedModels())
				.then(() => modelsStore.ensureFirstModelSelected());
		}
	});

	let scrollEl = $state<HTMLDivElement | null>(null);
	// where each page below the top one was scrolled, to come back to it
	let scrollStack: number[] = [];

	// each opening starts on the first page
	$effect(() => {
		if (open) {
			pages.reset();
			scrollStack = [];
		}
	});

	function scrollTo(top: number) {
		void tick().then(() => scrollEl?.scrollTo({ top }));
	}

	// a pushed page starts at its top, and back returns to where the page below was left
	function openPage(next: Page) {
		if (next.id === SettingsSheetPage.TOOLS) toolsPanel.handleOpen();

		scrollStack.push(scrollEl?.scrollTop ?? 0);
		pages.push(next);
		scrollTo(0);
	}

	function goBack() {
		pages.pop();
		scrollTo(scrollStack.pop() ?? 0);
	}

	/** Saves one change as it is made, checked and coupled as the desktop's Save does it. */
	function commit(key: string, value: string | boolean) {
		const updates: Record<string, string | number | boolean> = { [key]: value };

		if (key === SETTINGS_KEYS.CUSTOM_JSON && typeof value === 'string' && value.trim()) {
			try {
				JSON.parse(value);
			} catch {
				toast.error(SETTINGS_SHEET_TEXT.CUSTOM_JSON_INVALID);

				return;
			}
		}

		if (NUMERIC_FIELDS.includes(key) && value !== '') {
			const numeric = Number(value);
			const field = allFields.find((entry) => entry.key === key);

			if (isNaN(numeric)) {
				toast.error(SETTINGS_SHEET_TEXT.NEEDS_NUMBER(field?.label ?? key));

				return;
			}

			if (POSITIVE_INTEGER_FIELDS.includes(key)) {
				updates[key] = Math.max(
					field?.min ?? POSITIVE_INTEGER_MIN,
					Math.min(field?.max ?? Number.POSITIVE_INFINITY, Math.round(numeric))
				);
			} else {
				updates[key] = numeric;
			}
		}

		// discover needs the hub metadata reads and opens from the Models page, as on the desktop
		if (key === SETTINGS_KEYS.ENABLE_DISCOVER_MODELS) {
			updates[SETTINGS_KEYS.USE_HUGGING_FACE_HUB] = value === true;

			if (value === true) updates[SETTINGS_KEYS.SHOW_MODELS_MANAGER] = true;
		}

		settingsStore.updateMultipleConfig(updates);
	}

	function setTheme(theme: string) {
		settingsStore.updateTheme(theme);
	}

	function handleConfirmReset() {
		settingsStore.forceSyncWithServerDefaults();
		setMode(settingsStore.config.theme as ColorMode);
		showResetDialog = false;
		toast.success(SETTINGS_SHEET_TEXT.RESET_DONE);
	}
</script>

<!-- full width over the sidebar, so it does not read as part of the page slid aside; the drag
     handle stays above the header, which lies over the page -->
<Drawer.Root bind:open shouldScaleBackground={false}>
	<Drawer.Content
		class="gap-0 overflow-hidden {SHEET_FULL_HEIGHT_CLASS} {SHEET_CONTENT_CLASS} [&>div:first-child]:relative [&>div:first-child]:z-20"
	>
		<SheetHeader canGoBack={pages.canGoBack} floating onBack={goBack} {title} />

		<Drawer.Description class="sr-only">{SETTINGS_SHEET_TEXT.DESCRIPTION}</Drawer.Description>

		<SheetPages
			bind:ref={scrollEl}
			class="gap-8 px-4 pt-16 pb-[max(1.5rem,var(--safe-area-bottom))]"
			pageKey={JSON.stringify(page)}
			{pages}
		>
			{#if page.id === SettingsSheetPage.ROOT}
				{@render rootPage()}
			{:else if page.id === SettingsSheetPage.SECTION}
				{#if section}
					{@render sectionPage(section.slug)}
				{/if}
			{:else if page.id === SettingsSheetPage.MCP_SERVERS}
				<SettingsMcpServers />
			{:else}
				<SettingsSheetTools onOpen={openPage} view={page} />
			{/if}
		</SheetPages>
	</Drawer.Content>
</Drawer.Root>

<AlertDialog.Root bind:open={showResetDialog}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>{SETTINGS_SHEET_TEXT.RESET_TITLE}</AlertDialog.Title>

			<AlertDialog.Description>{SETTINGS_SHEET_TEXT.RESET_DESCRIPTION}</AlertDialog.Description>
		</AlertDialog.Header>

		<AlertDialog.Footer>
			<AlertDialog.Cancel>{SETTINGS_SHEET_TEXT.CANCEL}</AlertDialog.Cancel>

			<AlertDialog.Action onclick={handleConfirmReset}>
				{SETTINGS_SHEET_TEXT.RESET_CONFIRM}
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>

{#snippet themeTile(theme: ColorMode)}
	{@const isDark = theme === ColorMode.DARK}

	{#if theme === ColorMode.SYSTEM}
		<span class="absolute inset-0 bg-neutral-50"></span>

		<span class="absolute inset-0 bg-neutral-900 [clip-path:polygon(100%_0,100%_100%,0_100%)]"
		></span>
	{:else}
		<span class={['absolute inset-0', isDark ? 'bg-neutral-900' : 'bg-neutral-50']}></span>
	{/if}

	<span
		class={[
			'absolute top-2.5 left-2.5 h-1 w-1/2 rounded-full',
			isDark ? 'bg-neutral-600' : 'bg-neutral-300'
		]}
	></span>

	<span
		class={[
			'absolute top-5 left-2.5 h-1 w-1/3 rounded-full',
			isDark ? 'bg-neutral-700' : 'bg-neutral-200'
		]}
	></span>
{/snippet}

{#snippet rootPage()}
	<InsetList class="grid grid-cols-3 gap-3 p-3" title={SETTINGS_SHEET_TEXT.APPEARANCE}>
		{#each SETTINGS_SHEET_THEMES as theme (theme.value)}
			{@const selected = config.theme === theme.value}

			<button
				aria-pressed={selected}
				class="flex flex-col items-center gap-2"
				onclick={() => setTheme(theme.value)}
				type="button"
			>
				<span
					class={[
						'relative block aspect-[4/3] w-full overflow-hidden rounded-xl border border-border/60',
						selected && 'ring-2 ring-(--action-primary) ring-offset-2 ring-offset-background'
					]}
				>
					{@render themeTile(theme.value)}
				</span>

				<span class={['text-caption', selected ? 'text-foreground' : 'text-muted-foreground']}>
					{theme.label}
				</span>
			</button>
		{/each}
	</InsetList>

	<InsetList caption={SETTINGS_SHEET_TEXT.SAVED_NOTE}>
		{#each sections as entry (entry.slug)}
			<InsetListRow
				chevron
				icon={entry.icon}
				onclick={() =>
					openPage(
						entry.slug === SETTINGS_SECTION_SLUGS.TOOLS
							? { id: SettingsSheetPage.TOOLS }
							: { id: SettingsSheetPage.SECTION, slug: entry.slug }
					)}
			>
				<span class="flex-1">{entry.title}</span>
			</InsetListRow>
		{/each}
	</InsetList>
{/snippet}

{#snippet sectionPage(slug: string)}
	{#if slug === SETTINGS_SECTION_SLUGS.IMPORT_EXPORT}
		<SettingsChatImportExportTab inset />
	{:else if slug === SETTINGS_SECTION_SLUGS.VOICE}
		<SettingsChatVoiceTab
			fields={sectionFields}
			inset
			localConfig={config}
			onConfigChange={commit}
		/>
	{:else}
		<SettingsChatFieldsInset
			fields={sectionFields}
			localConfig={config}
			onConfigChange={commit}
			onManageMcpServers={() => openPage({ id: SettingsSheetPage.MCP_SERVERS })}
			onThemeChange={setTheme}
		/>

		{#if slug === SETTINGS_SECTION_SLUGS.GENERAL}
			<InsetList title={SETTINGS_SHEET_TEXT.APP}>
				<InsetListRow icon={RefreshCw} onclick={() => window.location.reload()}>
					{SETTINGS_SHEET_TEXT.RELOAD_APP}
				</InsetListRow>

				{#if settingsStore.hasCustomSettings}
					<InsetListRow
						bodyClass="text-destructive"
						icon={RotateCcw}
						iconClass="text-destructive"
						onclick={() => (showResetDialog = true)}
					>
						{SETTINGS_SHEET_TEXT.RESET}
					</InsetListRow>
				{/if}
			</InsetList>
		{/if}
	{/if}
{/snippet}
