<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { Checkbox } from '@ggml-org/llama-ui/ui/checkbox';
	import { Input } from '@ggml-org/llama-ui/ui/input';
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import * as RadioGroup from '@ggml-org/llama-ui/ui/radio-group';
	import * as Select from '@ggml-org/llama-ui/ui/select';
	import { Textarea } from '@ggml-org/llama-ui/ui/textarea';
	import { FlaskConical, RotateCcw } from '@lucide/svelte';
	import McpLogo from '#lib/components/app/mcp/McpLogo.svelte';
	import { SettingsChatParameterSourceIndicator } from '#lib/components/app/settings/index.ts';
	import {
		ICON_CLASS_DEFAULT,
		ICON_CLASS_SM,
		MANAGE_MCP_SERVERS_LABEL,
		SETTING_CONFIG_INFO,
		SETTINGS_KEYS
	} from '#lib/constants/index.ts';
	import { SettingsFieldType } from '#lib/enums/settings.enums.ts';
	import { useSettingsFieldDefaults } from '#lib/hooks/use-settings-field-defaults.svelte.ts';
	import { settingsStore, uiStore } from '#lib/stores/index.ts';
	import { normalizeFloatingPoint } from '#lib/utils/precision.ts';
	import type { Component } from 'svelte';

	interface Props {
		fields: SettingsFieldConfig[];
		localConfig: SettingsConfigType;
		onConfigChange: (key: string, value: string | boolean) => void;
		onThemeChange?: (theme: string) => void;
	}

	let { fields, localConfig, onConfigChange, onThemeChange }: Props = $props();

	const fieldDefaults = useSettingsFieldDefaults(() => localConfig);
	let currentModelParams = $derived(fieldDefaults.serverDefaults);
</script>

{#each fields as field (field.key)}
	{#if !field.dependsOn || Boolean(localConfig[field.dependsOn])}
		<div class={field.dependsOn ? 'space-y-2 pl-6' : 'space-y-2'}>
			{#if field.type === SettingsFieldType.INPUT}
				{@const currentValue = String(localConfig[field.key] ?? '')}
				{@const isCustomRealTime = fieldDefaults.isCustom(field)}

				<div class="flex items-center gap-2">
					<Label class="flex items-center gap-1.5 text-label" for={field.key}>
						{field.label}

						{#if field.isExperimental}
							<FlaskConical class="size-icon-sm text-muted-foreground" />
						{/if}
					</Label>

					{#if isCustomRealTime}
						<SettingsChatParameterSourceIndicator />
					{/if}
				</div>

				<div class="relative w-full">
					<Input
						autocomplete={field.isPrivate ? 'new-password' : 'off'}
						id={field.key}
						type={field.isPrivate ? 'password' : field.isPositiveInteger ? 'number' : 'text'}
						{...field.isPositiveInteger
							? {
									min: String(field.min ?? 1),
									step: '1',
									...(field.max != null ? { max: String(field.max) } : {})
								}
							: {}}
						class="w-full {isCustomRealTime ? 'pr-8' : ''}"
						oninput={(e) => onConfigChange(field.key, e.currentTarget.value)}
						placeholder={currentModelParams[field.key] != null
							? `Default: ${normalizeFloatingPoint(currentModelParams[field.key])}`
							: (field.placeholder ?? '')}
						value={currentValue}
					/>

					{#if isCustomRealTime}
						<button
							aria-label="Reset to default"
							class="absolute top-1/2 right-2 inline-flex size-5 -translate-y-1/2 items-center justify-center rounded transition-colors hover:bg-muted"
							onclick={() => {
								settingsStore.resetParameterToServerDefault(field.key);
								onConfigChange(field.key, '');
							}}
							title="Reset to default"
							type="button"
						>
							<RotateCcw class="size-icon-xs" />
						</button>
					{/if}
				</div>

				{#if field.help || SETTING_CONFIG_INFO[field.key]}
					<p class="mt-1 text-caption text-muted-foreground">
						{@html field.help || SETTING_CONFIG_INFO[field.key]}
					</p>
				{/if}
			{:else if field.type === SettingsFieldType.TEXTAREA}
				{#if field.label}
					<Label class="block flex items-center gap-1.5 text-label" for={field.key}>
						{field.label}

						{#if field.isExperimental}
							<FlaskConical class="size-icon-sm text-muted-foreground" />
						{/if}
					</Label>
				{/if}

				<Textarea
					class="min-h-40 w-full md:max-w-3xl"
					id={field.key}
					oninput={(e) => onConfigChange(field.key, e.currentTarget.value)}
					placeholder={field.placeholder ?? ''}
					value={String(localConfig[field.key] ?? '')}
				/>

				{#if field.help || SETTING_CONFIG_INFO[field.key]}
					<p class="mt-1 text-caption text-muted-foreground">
						{field.help || SETTING_CONFIG_INFO[field.key]}
					</p>
				{/if}

				{#if field.key === SETTINGS_KEYS.SYSTEM_MESSAGE}
					<div class="mt-3 flex items-center gap-2">
						<Checkbox
							checked={Boolean(localConfig.showSystemMessage ?? true)}
							id="showSystemMessage"
							onCheckedChange={(checked) =>
								onConfigChange(SETTINGS_KEYS.SHOW_SYSTEM_MESSAGE, Boolean(checked))}
						/>

						<Label class="cursor-pointer text-body-sm font-normal" for="showSystemMessage">
							Show system message in conversations
						</Label>
					</div>
				{/if}
			{:else if field.type === SettingsFieldType.SELECT}
				{@const selectedOption = field.options?.find(
					(opt: { value: string; label: string; icon?: Component }) =>
						opt.value === localConfig[field.key]
				)}
				{@const currentValue = localConfig[field.key]}
				{@const isCustomRealTime = fieldDefaults.isCustom(field)}

				<div class="flex items-center gap-2">
					<Label class="flex items-center gap-1.5 text-label" for={field.key}>
						{field.label}

						{#if field.isExperimental}
							<FlaskConical class="size-icon-sm text-muted-foreground" />
						{/if}
					</Label>

					{#if isCustomRealTime}
						<SettingsChatParameterSourceIndicator />
					{/if}
				</div>

				<Select.Root
					onValueChange={(value) => {
						if (field.key === SETTINGS_KEYS.THEME && value && onThemeChange) {
							onThemeChange(value);
						} else {
							onConfigChange(field.key, value);
						}
					}}
					type="single"
					value={currentValue}
				>
					<div class="relative w-full md:w-auto">
						<Select.Trigger class="w-full">
							<div class="flex items-center gap-2">
								{#if selectedOption?.icon}
									{@const IconComponent = selectedOption.icon}
									<IconComponent class={ICON_CLASS_SM} />
								{/if}

								{selectedOption?.label || `Select ${field.label.toLowerCase()}`}
							</div>
						</Select.Trigger>

						{#if isCustomRealTime}
							<button
								aria-label="Reset to default"
								class="absolute top-1/2 right-8 inline-flex size-5 -translate-y-1/2 items-center justify-center rounded transition-colors hover:bg-muted"
								onclick={() => {
									settingsStore.resetParameterToServerDefault(field.key);
									onConfigChange(field.key, '');
								}}
								title="Reset to default"
								type="button"
							>
								<RotateCcw class="size-icon-xs" />
							</button>
						{/if}
					</div>

					<Select.Content>
						{#if field.options}
							{#each field.options as option (option.value)}
								<Select.Item label={option.label} value={option.value}>
									<div class="flex items-center gap-2">
										{#if option.icon}
											{@const IconComponent = option.icon}
											<IconComponent class={ICON_CLASS_DEFAULT} />
										{/if}
										{option.label}
									</div>
								</Select.Item>
							{/each}
						{/if}
					</Select.Content>
				</Select.Root>

				{#if field.help || SETTING_CONFIG_INFO[field.key]}
					<p class="mt-1 text-caption text-muted-foreground">
						{field.help || SETTING_CONFIG_INFO[field.key]}
					</p>
				{/if}
			{:else if field.type === SettingsFieldType.RADIO && field.radioOptions}
				{@const radioOptions = field.radioOptions}
				{@const currentMode =
					radioOptions.find((o: { key: string }) => Boolean(localConfig[o.key]))?.value ??
					radioOptions[0].value}

				<Label class="flex items-center gap-1.5 text-label mb-4">
					{field.label}

					{#if field.isExperimental}
						<FlaskConical class="size-icon-sm text-muted-foreground" />
					{/if}
				</Label>

				<RadioGroup.Root
					class="gap-4"
					onValueChange={(value) => {
						for (const opt of radioOptions) {
							onConfigChange(opt.key, opt.value === value);
						}
					}}
					value={currentMode}
				>
					{#each radioOptions as opt (opt.value)}
						{@const itemId = `${field.key}-${opt.value}`}
						<div class="flex items-center gap-2">
							<RadioGroup.Item id={itemId} value={opt.value} />

							<Label
								class="flex cursor-pointer items-center gap-1.5 text-body-sm font-normal"
								for={itemId}
							>
								{opt.label}

								{#if opt.isExperimental}
									<FlaskConical class="size-icon-sm text-muted-foreground" />
								{/if}
							</Label>
						</div>
					{/each}
				</RadioGroup.Root>

				{#if field.help || SETTING_CONFIG_INFO[field.key]}
					<p class="text-caption text-muted-foreground">
						{field.help || SETTING_CONFIG_INFO[field.key]}
					</p>
				{/if}
			{:else if field.type === SettingsFieldType.CHECKBOX}
				<div class="flex items-start space-x-3">
					<Checkbox
						checked={Boolean(localConfig[field.key])}
						class="mt-1"
						disabled={fieldDefaults.isLocked(field.key)}
						id={field.key}
						onCheckedChange={(checked) => onConfigChange(field.key, checked)}
					/>

					<div class="space-y-1">
						<label
							class="flex items-center gap-1.5 pt-1 pb-0.5 text-label leading-none {fieldDefaults.isLocked(
								field.key
							)
								? 'cursor-default opacity-80'
								: 'cursor-pointer'}"
							for={field.key}
						>
							{field.label}

							{#if field.isExperimental}
								<FlaskConical class="size-icon-sm text-muted-foreground" />
							{/if}
						</label>

						{#if field.help || SETTING_CONFIG_INFO[field.key]}
							<p class="text-caption text-muted-foreground">
								{field.help || SETTING_CONFIG_INFO[field.key]}
							</p>
						{/if}

						{#if field.key === SETTINGS_KEYS.ENABLE_MCP && Boolean(localConfig[field.key])}
							<Button
								class="mt-2"
								onclick={() => uiStore.openMcpServers()}
								size="sm"
								variant="outline"
							>
								<McpLogo class="size-icon-md" />

								{MANAGE_MCP_SERVERS_LABEL}
							</Button>
						{/if}
					</div>
				</div>
			{/if}
		</div>
	{/if}
{/each}
