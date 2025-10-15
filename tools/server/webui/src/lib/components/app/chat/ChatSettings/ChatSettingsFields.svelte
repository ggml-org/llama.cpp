<script lang="ts">
	import { RotateCcw } from '@lucide/svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import * as Select from '$lib/components/ui/select';
	import { Textarea } from '$lib/components/ui/textarea';
	import { SETTING_CONFIG_DEFAULT, SETTING_CONFIG_INFO } from '$lib/constants/settings-config';
	import { supportsVision } from '$lib/stores/server.svelte';
	import { getParameterInfo, resetParameterToServerDefault } from '$lib/stores/settings.svelte';
	import { ParameterSyncService } from '$lib/services/parameter-sync';
	import ParameterSourceIndicator from './ParameterSourceIndicator.svelte';
	import type { Component } from 'svelte';

	interface Props {
		fields: SettingsFieldConfig[];
		localConfig: SettingsConfigType;
		onConfigChange: (key: string, value: string | boolean) => void;
		onThemeChange?: (theme: string) => void;
	}

	let { fields, localConfig, onConfigChange, onThemeChange }: Props = $props();

	// Helper function to get parameter source info for syncable parameters
	function getParameterSourceInfo(key: string) {
		if (!ParameterSyncService.canSyncParameter(key)) {
			return null;
		}

		return getParameterInfo(key);
	}
</script>

{#each fields as field (field.key)}
	<div class="space-y-2">
		{#if field.type === 'input'}
			{@const paramInfo = getParameterSourceInfo(field.key)}
			{@const currentValue = String(localConfig[field.key] ?? '')}
			{@const propsDefault = paramInfo?.serverDefault}
			{@const isCustomRealTime = (() => {
				if (!paramInfo || propsDefault === undefined) return false;

				// Apply same rounding logic for real-time comparison
				const inputValue = currentValue;
				const numericInput = parseFloat(inputValue);
				const normalizedInput = !isNaN(numericInput)
					? Math.round(numericInput * 1000000) / 1000000
					: inputValue;
				const normalizedDefault =
					typeof propsDefault === 'number'
						? Math.round(propsDefault * 1000000) / 1000000
						: propsDefault;

				return normalizedInput !== normalizedDefault;
			})()}

			<div class="flex items-center gap-2">
				<Label for={field.key} class="text-sm font-medium">
					{field.label}
				</Label>
				{#if isCustomRealTime}
					<ParameterSourceIndicator />
				{/if}
			</div>

			<div class="relative w-full md:max-w-md">
				<Input
					id={field.key}
					value={currentValue}
					oninput={(e) => {
						// Update local config immediately for real-time badge feedback
						onConfigChange(field.key, e.currentTarget.value);
					}}
					placeholder={`Default: ${SETTING_CONFIG_DEFAULT[field.key] ?? 'none'}`}
					class="w-full {isCustomRealTime ? 'pr-8' : ''}"
				/>
				{#if isCustomRealTime}
					<button
						type="button"
						onclick={() => {
							resetParameterToServerDefault(field.key);
							// Trigger UI update by calling onConfigChange with the default value
							const defaultValue = propsDefault ?? SETTING_CONFIG_DEFAULT[field.key];
							onConfigChange(field.key, String(defaultValue));
						}}
						class="absolute top-1/2 right-2 inline-flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded transition-colors hover:bg-muted"
						aria-label="Reset to default"
						title="Reset to default"
					>
						<RotateCcw class="h-3 w-3" />
					</button>
				{/if}
			</div>
			{#if field.help || SETTING_CONFIG_INFO[field.key]}
				<p class="mt-1 text-xs text-muted-foreground">
					{field.help || SETTING_CONFIG_INFO[field.key]}
				</p>
			{/if}
		{:else if field.type === 'textarea'}
			<Label for={field.key} class="block text-sm font-medium">
				{field.label}
			</Label>

			<Textarea
				id={field.key}
				value={String(localConfig[field.key] ?? '')}
				onchange={(e) => onConfigChange(field.key, e.currentTarget.value)}
				placeholder={`Default: ${SETTING_CONFIG_DEFAULT[field.key] ?? 'none'}`}
				class="min-h-[100px] w-full md:max-w-2xl"
			/>
			{#if field.help || SETTING_CONFIG_INFO[field.key]}
				<p class="mt-1 text-xs text-muted-foreground">
					{field.help || SETTING_CONFIG_INFO[field.key]}
				</p>
			{/if}
		{:else if field.type === 'select'}
			{@const selectedOption = field.options?.find(
				(opt: { value: string; label: string; icon?: Component }) =>
					opt.value === localConfig[field.key]
			)}
			{@const paramInfo = getParameterSourceInfo(field.key)}
			{@const currentValue = localConfig[field.key]}
			{@const propsDefault = paramInfo?.serverDefault}
			{@const isCustomRealTime = (() => {
				if (!paramInfo || propsDefault === undefined) return false;

				// For select fields, do direct comparison (no rounding needed)
				return currentValue !== propsDefault;
			})()}

			<div class="flex items-center gap-2">
				<Label for={field.key} class="text-sm font-medium">
					{field.label}
				</Label>
				{#if isCustomRealTime}
					<ParameterSourceIndicator />
				{/if}
			</div>

			<Select.Root
				type="single"
				value={currentValue}
				onValueChange={(value) => {
					if (field.key === 'theme' && value && onThemeChange) {
						onThemeChange(value);
					} else {
						onConfigChange(field.key, value);
					}
				}}
			>
				<div class="relative w-full md:w-auto md:max-w-md">
					<Select.Trigger class="w-full">
						<div class="flex items-center gap-2">
							{#if selectedOption?.icon}
								{@const IconComponent = selectedOption.icon}
								<IconComponent class="h-4 w-4" />
							{/if}

							{selectedOption?.label || `Select ${field.label.toLowerCase()}`}
						</div>
					</Select.Trigger>
					{#if isCustomRealTime}
						<button
							type="button"
							onclick={() => {
								resetParameterToServerDefault(field.key);
								// Trigger UI update by calling onConfigChange with the default value
								const defaultValue = propsDefault ?? SETTING_CONFIG_DEFAULT[field.key];
								onConfigChange(field.key, String(defaultValue));
							}}
							class="absolute top-1/2 right-8 inline-flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded transition-colors hover:bg-muted"
							aria-label="Reset to default"
							title="Reset to default"
						>
							<RotateCcw class="h-3 w-3" />
						</button>
					{/if}
				</div>
				<Select.Content>
					{#if field.options}
						{#each field.options as option (option.value)}
							<Select.Item value={option.value} label={option.label}>
								<div class="flex items-center gap-2">
									{#if option.icon}
										{@const IconComponent = option.icon}
										<IconComponent class="h-4 w-4" />
									{/if}
									{option.label}
								</div>
							</Select.Item>
						{/each}
					{/if}
				</Select.Content>
			</Select.Root>
			{#if field.help || SETTING_CONFIG_INFO[field.key]}
				<p class="mt-1 text-xs text-muted-foreground">
					{field.help || SETTING_CONFIG_INFO[field.key]}
				</p>
			{/if}
		{:else if field.type === 'checkbox'}
			{@const isDisabled = field.key === 'pdfAsImage' && !supportsVision()}

			<div class="flex items-start space-x-3">
				<Checkbox
					id={field.key}
					checked={Boolean(localConfig[field.key])}
					disabled={isDisabled}
					onCheckedChange={(checked) => onConfigChange(field.key, checked)}
					class="mt-1"
				/>

				<div class="space-y-1">
					<label
						for={field.key}
						class="cursor-pointer text-sm leading-none font-medium {isDisabled
							? 'text-muted-foreground'
							: ''}"
					>
						{field.label}
					</label>

					{#if field.help || SETTING_CONFIG_INFO[field.key]}
						<p class="text-xs text-muted-foreground">
							{field.help || SETTING_CONFIG_INFO[field.key]}
						</p>
					{:else if field.key === 'pdfAsImage' && !supportsVision()}
						<p class="text-xs text-muted-foreground">
							PDF-to-image processing requires a vision-capable model. PDFs will be processed as
							text.
						</p>
					{/if}
				</div>
			</div>
		{/if}
	</div>
{/each}
