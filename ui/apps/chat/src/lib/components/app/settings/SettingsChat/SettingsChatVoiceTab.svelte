<!--
	Voice section of the settings: the registry's fields, with the transcription model between them. That one lists the
	server's models with audio input, which a static field cannot, so it is drawn here.
-->
<script lang="ts">
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import * as Select from '@ggml-org/llama-ui/ui/select';
	import { SettingsChatFields } from '#lib/components/app/settings/index.ts';
	import { SETTING_CONFIG_INFO, SETTINGS_KEYS } from '#lib/constants/index.ts';
	import { modelsStore, serverStore } from '#lib/stores/index.ts';

	interface Props {
		fields: SettingsFieldConfig[];
		localConfig: SettingsConfigType;
		onConfigChange: (key: string, value: string | boolean) => void;
	}

	let { fields, localConfig, onConfigChange }: Props = $props();

	const KEY = SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL;
	const SAME_AS_CHAT = '';
	const SAME_AS_CHAT_LABEL = 'Same as chat model';

	// the picker sits right after the Voice input field
	let splitAt = $derived(fields.findIndex((f) => f.key === SETTINGS_KEYS.VOICE_INPUT_MODE) + 1);
	let isRouter = $derived(serverStore.isRouterMode);
	let current = $derived(String(localConfig[KEY] ?? SAME_AS_CHAT));

	let audioModels = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		return modelsStore
			.getVisibleModels()
			.filter((option) => modelsStore.props.modelSupportsAudio(option.model));
	});

	let options = $derived.by(() => {
		const list = audioModels.map((option) => ({ label: option.name, value: option.model }));

		// a model picked earlier that the server no longer lists stays visible as the choice
		if (current && !list.some((option) => option.value === current)) {
			list.push({ label: current, value: current });
		}

		return [{ label: SAME_AS_CHAT_LABEL, value: SAME_AS_CHAT }, ...list];
	});

	let selectedLabel = $derived(
		options.find((option) => option.value === current)?.label ?? SAME_AS_CHAT_LABEL
	);

	let singleModelId = $derived(modelsStore.activeModelId);
	// the file's name, not the GGUF path a single-model server reports
	let singleModel = $derived(
		singleModelId ? (singleModelId.split('/').pop() ?? singleModelId).replace(/\.gguf$/i, '') : null
	);
	let singleModelHasAudio = $derived(
		singleModelId ? modelsStore.props.modelSupportsAudio(singleModelId) : false
	);
</script>

<div class="space-y-6">
	<SettingsChatFields fields={fields.slice(0, splitAt)} {localConfig} {onConfigChange} />

	<div class="space-y-2" data-slot="voice-transcription-model">
		<Label class="text-sm font-medium" for={KEY}>Transcription model</Label>

		{#if isRouter}
			<Select.Root
				onValueChange={(value) => onConfigChange(KEY, value)}
				type="single"
				value={current}
			>
				<Select.Trigger class="w-full" id={KEY}>{selectedLabel}</Select.Trigger>

				<Select.Content>
					{#each options as option (option.value)}
						<Select.Item label={option.label} value={option.value}>{option.label}</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>

			<p class="mt-1 text-xs text-muted-foreground">
				{SETTING_CONFIG_INFO[KEY]} Only models with audio input are listed.
			</p>

			<p class="text-xs text-muted-foreground">
				A transcription model other than the chat model is loaded on the server when you dictate,
				alongside the chat model. A server that keeps one model at a time (<code
					>--models-max 1</code
				>) unloads the chat model to do it.
			</p>
		{:else}
			<p class="text-sm text-muted-foreground">
				{#if singleModel}
					This server runs one model, <span class="font-medium text-foreground">{singleModel}</span
					>.
					{singleModelHasAudio
						? 'It has audio input, so it transcribes your voice.'
						: 'It has no audio input, so voice input is not available on this server.'}
				{:else}
					This server runs one model, which transcribes your voice when it has audio input.
				{/if}
			</p>
		{/if}
	</div>

	<SettingsChatFields fields={fields.slice(splitAt)} {localConfig} {onConfigChange} />
</div>
