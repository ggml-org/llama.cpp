<script lang="ts">
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Input } from '$lib/components/ui/input';
	import { SettingsFieldType } from '$lib/enums/settings.enums';
	import { modelsStore, serverStore, settingsStore } from '$lib/stores';
	import type { SettingsSection } from '$lib/types';
	import { normalizeFloatingPoint } from '$lib/utils/precision';

	interface Props {
		section: SettingsSection;
	}

	let { section }: Props = $props();

	let currentModelParams = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		if (serverStore.isRouterMode) {
			const currentModelName = modelsStore.selectedModelName;

			if (currentModelName) {
				const currentModelProps = modelsStore.props.getModelProps(currentModelName);

				return (currentModelProps?.default_generation_settings?.params ?? {}) as Record<
					string,
					unknown
				>;
			}
		}

		return (serverStore.defaultParams ?? {}) as Record<string, unknown>;
	});

	function currentValue(key: string): string {
		const value = settingsStore.config[key];

		return value == null ? '' : String(value);
	}

	function handleInput(key: string, value: string) {
		settingsStore.updateConfig(key, value);
	}

	function placeholder(key: string): string {
		const serverDefault = currentModelParams[key];

		return serverDefault != null ? `Default: ${normalizeFloatingPoint(serverDefault)}` : '';
	}
</script>

<div class="grid gap-1">
	{#each section.fields ?? [] as field (field.key)}
		{#if field.type === SettingsFieldType.INPUT}
			<label class="mb-2 flex flex-col gap-1.5">
				<span class="text-xs font-medium text-muted-foreground">{field.label}</span>

				<Input
					type="text"
					value={currentValue(field.key)}
					placeholder={placeholder(field.key)}
					oninput={(event) => handleInput(field.key, event.currentTarget.value)}
					class="h-8"
				/>
			</label>
		{:else if field.type === SettingsFieldType.CHECKBOX}
			<label
				class="flex cursor-pointer items-center gap-2 rounded-md px-1 py-1.5 text-sm hover:bg-accent"
			>
				<Checkbox
					checked={Boolean(settingsStore.config[field.key])}
					onCheckedChange={(checked) => settingsStore.updateConfig(field.key, Boolean(checked))}
				/>

				<span>{field.label}</span>
			</label>
		{/if}
	{/each}
</div>
