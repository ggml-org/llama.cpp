<!--
	The settings fields as a phone's native inset list: switches, pickers and numbers as rows of
	one group, prompts and choices of one as groups of their own, each help text under its label.
	A change is handed on when it is made: a switch or a pick at once, typed text once it is left.
-->
<script lang="ts">
	import { NativeSelect } from '@ggml-org/llama-ui/app/forms';
	import { InsetList, InsetListRow } from '@ggml-org/llama-ui/app/sheets';
	import { Switch } from '@ggml-org/llama-ui/ui/switch';
	import { Textarea } from '@ggml-org/llama-ui/ui/textarea';
	import { Check, FlaskConical, RotateCcw } from '@lucide/svelte';
	import McpLogo from '#lib/components/app/mcp/McpLogo.svelte';
	import {
		MANAGE_MCP_SERVERS_LABEL,
		NUMERIC_FIELDS,
		SETTING_CONFIG_INFO,
		SETTINGS_KEYS,
		SETTINGS_SHEET_TEXT
	} from '#lib/constants/index.ts';
	import { SettingsFieldBlock, SettingsFieldType } from '#lib/enums/settings.enums.ts';
	import { useSettingsFieldDefaults } from '#lib/hooks/use-settings-field-defaults.svelte.ts';
	import { settingsStore, uiStore } from '#lib/stores/index.ts';
	import { normalizeFloatingPoint } from '#lib/utils/precision.ts';
	import { tick } from 'svelte';

	interface Props {
		/** The subgroup's title, shown above its first group of rows. */
		title?: string;
		fields: SettingsFieldConfig[];
		localConfig: SettingsConfigType;
		onConfigChange: (key: string, value: string | boolean) => void;
		/** Opens the MCP servers as a page of the sheet; without it, they open in their dialog */
		onManageMcpServers?: () => void;
		onThemeChange?: (theme: string) => void;
	}

	let { fields, localConfig, onConfigChange, onManageMcpServers, onThemeChange, title }: Props =
		$props();

	/** Rows share a group; a prompt or a choice of one gets a group of its own. */
	type Block =
		| { fields: SettingsFieldConfig[]; kind: SettingsFieldBlock.ROWS }
		| { field: SettingsFieldConfig; kind: SettingsFieldBlock.OWN };

	const fieldDefaults = useSettingsFieldDefaults(() => localConfig);

	let visibleFields = $derived(
		fields.filter((field) => !field.dependsOn || Boolean(localConfig[field.dependsOn]))
	);

	let blocks = $derived.by(() => {
		const result: Block[] = [];

		for (const field of visibleFields) {
			const ownGroup =
				field.type === SettingsFieldType.TEXTAREA ||
				field.type === SettingsFieldType.RADIO ||
				(field.type === SettingsFieldType.INPUT && !isNumeric(field));

			if (ownGroup) {
				result.push({ field, kind: SettingsFieldBlock.OWN });

				continue;
			}

			const last = result[result.length - 1];

			if (last?.kind === SettingsFieldBlock.ROWS) last.fields.push(field);
			else result.push({ fields: [field], kind: SettingsFieldBlock.ROWS });
		}

		return result;
	});

	function isNumeric(field: SettingsFieldConfig): boolean {
		return field.isPositiveInteger || NUMERIC_FIELDS.includes(field.key);
	}

	// a prompt alone in a subgroup of its own name needs the name only once
	function ownCaption(field: SettingsFieldConfig): string | null {
		return title && field.label.toLowerCase() === title.toLowerCase() ? null : field.label;
	}

	function helpOf(field: SettingsFieldConfig): string | undefined {
		return field.help || SETTING_CONFIG_INFO[field.key] || undefined;
	}

	function valueOf(key: string): string {
		return String(localConfig[key] ?? '');
	}

	// typed text is handed on once the field is left; a value that was not taken (not a number,
	// invalid JSON) or was adjusted gives way to the saved one
	function handleTextChange(
		event: Event & { currentTarget: HTMLInputElement | HTMLTextAreaElement }
	) {
		const input = event.currentTarget;
		const key = input.name;

		onConfigChange(key, input.value);
		void tick().then(() => (input.value = valueOf(key)));
	}

	function resetToDefault(key: string) {
		settingsStore.resetParameterToServerDefault(key);
		onConfigChange(key, '');
	}
</script>

{#if blocks.length > 0}
	<div class="flex flex-col gap-1.5">
		{#if title}
			<h3 class="inset-list-title">{title}</h3>
		{/if}

		<div class="flex flex-col gap-6">
			{#each blocks as block, index (block.kind === SettingsFieldBlock.OWN ? block.field.key : `${block.kind}-${index}`)}
				{#if block.kind === SettingsFieldBlock.ROWS}
					<InsetList>
						{#each block.fields as field (field.key)}
							{#if field.type === SettingsFieldType.CHECKBOX}
								{@render switchRow(field)}
							{:else if field.type === SettingsFieldType.SELECT}
								{@render selectRow(field)}
							{:else}
								{@render numberRow(field)}
							{/if}

							{#if field.key === SETTINGS_KEYS.ENABLE_MCP && Boolean(localConfig[field.key])}
								<InsetListRow
									chevron
									onclick={() =>
										onManageMcpServers ? onManageMcpServers() : uiStore.openMcpServers()}
								>
									{#snippet leading()}
										<McpLogo class="size-icon-md shrink-0" />
									{/snippet}

									<span class="flex-1">{MANAGE_MCP_SERVERS_LABEL}</span>
								</InsetListRow>
							{/if}
						{/each}
					</InsetList>
				{:else}
					{@const field = block.field}
					{@const help = helpOf(field)}
					{@const showCaption = index > 0 || ownCaption(field) !== null}

					<div class="flex flex-col gap-1.5">
						{#if showCaption}
							<h3
								class={[
									title ? 'inset-list-caption' : 'inset-list-title',
									'flex items-center gap-1.5'
								]}
							>
								{field.label}

								{#if field.isExperimental}
									<FlaskConical class="size-icon-sm" />
								{/if}
							</h3>
						{/if}

						<InsetList>
							{#if field.type === SettingsFieldType.TEXTAREA}
								<Textarea
									aria-label={field.label}
									class="min-h-32 resize-none rounded-none border-0! bg-transparent! px-4 py-3 text-body shadow-none! focus-visible:ring-0 dark:bg-transparent!"
									name={field.key}
									onchange={handleTextChange}
									placeholder={field.placeholder ?? ''}
									value={valueOf(field.key)}
								/>

								{#if field.key === SETTINGS_KEYS.SYSTEM_MESSAGE}
									<InsetListRow as="label" bodyClass="gap-4 py-3">
										<span class="flex-1">{SETTINGS_SHEET_TEXT.SHOW_SYSTEM_MESSAGE}</span>

										<Switch
											checked={Boolean(localConfig[SETTINGS_KEYS.SHOW_SYSTEM_MESSAGE])}
											onCheckedChange={(checked) =>
												onConfigChange(SETTINGS_KEYS.SHOW_SYSTEM_MESSAGE, checked)}
											size="lg"
										/>
									</InsetListRow>
								{/if}
							{:else if field.type === SettingsFieldType.RADIO && field.radioOptions}
								{@const radioOptions = field.radioOptions}
								{@const currentMode =
									radioOptions.find((option: { key: string; value: string }) =>
										Boolean(localConfig[option.key])
									)?.value ?? radioOptions[0].value}

								{#each radioOptions as option (option.value)}
									<!-- an option's name can run to two lines, so the row grows with it -->
									<InsetListRow
										aria-pressed={option.value === currentMode}
										bodyClass="gap-4 py-3"
										onclick={() => {
											for (const opt of radioOptions)
												onConfigChange(opt.key, opt.value === option.value);
										}}
									>
										<span class="flex flex-1 items-center gap-1.5 leading-snug">
											{option.label}

											{#if option.isExperimental}
												<FlaskConical class="size-icon-sm text-muted-foreground" />
											{/if}
										</span>

										{#if option.value === currentMode}
											<Check class="size-4 text-(--action-primary)" />
										{/if}
									</InsetListRow>
								{/each}
							{:else}
								<!-- text: a key, a name, a language; typed on a line of its own -->
								<input
									aria-label={field.label}
									autocomplete={field.isPrivate ? 'new-password' : 'off'}
									class="min-h-12 w-full border-0 bg-transparent px-4 text-body shadow-none outline-none placeholder:text-muted-foreground focus:ring-0"
									name={field.key}
									onchange={handleTextChange}
									placeholder={field.placeholder ?? ''}
									type={field.isPrivate ? 'password' : 'text'}
									value={valueOf(field.key)}
								/>
							{/if}
						</InsetList>

						{#if help}
							<!-- eslint-disable-next-line svelte/no-at-html-tags -- help text comes from the settings registry -->
							<p class="inset-list-caption">{@html help}</p>
						{/if}
					</div>
				{/if}
			{/each}
		</div>
	</div>
{/if}

{#snippet rowLabel(field: SettingsFieldConfig)}
	{@const help = helpOf(field)}

	<span class="flex min-w-0 flex-1 flex-col gap-0.5 py-3">
		<span class="flex items-center gap-1.5">
			{field.label}

			{#if field.isExperimental}
				<FlaskConical class="size-icon-sm shrink-0 text-muted-foreground" />
			{/if}
		</span>

		{#if help}
			<!-- eslint-disable-next-line svelte/no-at-html-tags -- help text comes from the settings registry -->
			<span class="text-caption text-muted-foreground">{@html help}</span>
		{/if}
	</span>
{/snippet}

{#snippet switchRow(field: SettingsFieldConfig)}
	{@const locked = fieldDefaults.isLocked(field.key)}

	<InsetListRow
		as="label"
		bodyClass={['gap-4', field.dependsOn && 'pl-4']}
		class={locked ? 'cursor-default opacity-60' : undefined}
	>
		{@render rowLabel(field)}

		<Switch
			checked={Boolean(localConfig[field.key])}
			disabled={locked}
			onCheckedChange={(checked) => onConfigChange(field.key, checked)}
			size="lg"
		/>
	</InsetListRow>
{/snippet}

{#snippet selectRow(field: SettingsFieldConfig)}
	<InsetListRow as="div" bodyClass="gap-4">
		{@render rowLabel(field)}

		<NativeSelect
			label={field.label}
			onValueChange={(value) => {
				if (field.key === SETTINGS_KEYS.THEME && value && onThemeChange) onThemeChange(value);
				else onConfigChange(field.key, value);
			}}
			options={(field.options ?? []).map((option: { value: string; label: string }) => ({
				label: option.label,
				value: option.value
			}))}
			value={valueOf(field.key)}
			variant="inset"
		/>
	</InsetListRow>
{/snippet}

{#snippet numberRow(field: SettingsFieldConfig)}
	{@const custom = fieldDefaults.isCustom(field)}
	{@const serverDefault = fieldDefaults.serverDefaults[field.key]}

	<InsetListRow as="div" bodyClass="gap-3">
		{@render rowLabel(field)}

		{#if custom}
			<button
				aria-label={SETTINGS_SHEET_TEXT.RESET_FIELD(field.label)}
				class="flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground active:bg-(--surface-pressed)"
				onclick={() => resetToDefault(field.key)}
				type="button"
			>
				<RotateCcw class="size-3.5" />
			</button>
		{/if}

		<!-- 16px at least, or iOS zooms the page in on focus -->
		<input
			aria-label={field.label}
			autocomplete="off"
			class={[
				'h-9 w-20 shrink-0 rounded-lg border-0 bg-(--surface-muted) px-2.5 text-right text-body tabular-nums shadow-none outline-none placeholder:text-muted-foreground focus:ring-1 focus:ring-(--line-focus)',
				custom && 'text-foreground'
			]}
			inputmode="decimal"
			name={field.key}
			onchange={handleTextChange}
			placeholder={serverDefault != null
				? String(normalizeFloatingPoint(serverDefault))
				: (field.placeholder ?? '')}
			type="text"
			value={valueOf(field.key)}
		/>
	</InsetListRow>
{/snippet}
