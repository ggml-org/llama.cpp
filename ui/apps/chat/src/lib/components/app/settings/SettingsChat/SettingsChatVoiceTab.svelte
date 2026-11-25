<!--
	Voice section of the settings: the registry's fields, with the microphone and the transcription model between them.
	Those list this device's microphones and the server's models with audio input, which a static field cannot, so they
	are drawn here.
-->
<script lang="ts">
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import * as Select from '@ggml-org/llama-ui/ui/select';
	import {
		listMicrophones,
		type Microphone,
		VOICE_MICROPHONE_AUTO,
		VOICE_MICROPHONE_DEFAULT
	} from '@ggml-org/llama-voice';
	import { SettingsChatFields } from '#lib/components/app/settings/index.ts';
	import {
		SETTING_CONFIG_INFO,
		SETTINGS_KEYS,
		VOICE_GGUF_EXTENSION_PATTERN,
		VOICE_SETTINGS_LABEL,
		VOICE_TRANSCRIPTION_MODEL_AUTO
	} from '#lib/constants/index.ts';
	import { modelsStore, serverStore } from '#lib/stores/index.ts';
	import { isTranscriptionOnlyModel } from '#lib/utils/voice-mode.ts';
	import { onMount } from 'svelte';

	interface Props {
		fields: SettingsFieldConfig[];
		localConfig: SettingsConfigType;
		onConfigChange: (key: string, value: string | boolean) => void;
	}

	let { fields, localConfig, onConfigChange }: Props = $props();

	const KEY = SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL;

	// the picker sits right after the Voice input field
	let splitAt = $derived(fields.findIndex((f) => f.key === SETTINGS_KEYS.VOICE_INPUT_MODE) + 1);
	let isRouter = $derived(serverStore.isRouterMode);
	let current = $derived(String(localConfig[KEY] ?? VOICE_TRANSCRIPTION_MODEL_AUTO));

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

		return [
			{ label: VOICE_SETTINGS_LABEL.automatic, value: VOICE_TRANSCRIPTION_MODEL_AUTO },
			...list
		];
	});

	// speech-to-text models first: they are made for dictation, the chat models that hear can do it too
	let optionGroups = $derived.by(() => {
		const models = options.filter((option) => option.value !== VOICE_TRANSCRIPTION_MODEL_AUTO);

		return [
			{
				heading: VOICE_SETTINGS_LABEL.speechToText,
				items: models.filter((option) => isTranscriptionOnlyModel(option.value))
			},
			{
				heading: VOICE_SETTINGS_LABEL.chatModels,
				items: models.filter((option) => !isTranscriptionOnlyModel(option.value))
			}
		].filter((group) => group.items.length > 0);
	});

	let selectedLabel = $derived(
		options.find((option) => option.value === current)?.label ?? VOICE_SETTINGS_LABEL.automatic
	);

	const MIC_KEY = SETTINGS_KEYS.VOICE_MICROPHONE;

	let microphones = $state<Microphone[]>([]);
	let micCurrent = $derived(String(localConfig[MIC_KEY] ?? VOICE_MICROPHONE_AUTO));
	// the browser names its microphones only once the page may use one
	let micNamesHidden = $derived(microphones.some((mic) => !mic.label));
	let micOptions = $derived.by(() => {
		const list = microphones.map((mic, index) => ({
			label: mic.label || VOICE_SETTINGS_LABEL.microphone(index + 1),
			value: mic.deviceId
		}));
		const fixed = [VOICE_MICROPHONE_AUTO, VOICE_MICROPHONE_DEFAULT];

		// a microphone picked earlier that is not plugged in now stays visible as the choice
		if (!fixed.includes(micCurrent) && !list.some((option) => option.value === micCurrent)) {
			list.push({ label: VOICE_SETTINGS_LABEL.disconnectedMicrophone, value: micCurrent });
		}

		return [
			{ label: VOICE_SETTINGS_LABEL.automatic, value: VOICE_MICROPHONE_AUTO },
			{ label: VOICE_SETTINGS_LABEL.systemDefault, value: VOICE_MICROPHONE_DEFAULT },
			...list.filter((option) => option.value)
		];
	});
	let micLabel = $derived(
		micOptions.find((option) => option.value === micCurrent)?.label ??
			VOICE_SETTINGS_LABEL.automatic
	);

	onMount(() => {
		const devices = navigator.mediaDevices;
		const refresh = () => {
			listMicrophones().then(
				(list) => (microphones = list),
				() => (microphones = [])
			);
		};

		refresh();
		devices?.addEventListener('devicechange', refresh);

		return () => devices?.removeEventListener('devicechange', refresh);
	});

	let singleModelId = $derived(modelsStore.activeModelId);
	// the file's name, not the GGUF path a single-model server reports
	let singleModel = $derived(
		singleModelId
			? (singleModelId.split('/').pop() ?? singleModelId).replace(VOICE_GGUF_EXTENSION_PATTERN, '')
			: null
	);
	let singleModelHasAudio = $derived(
		singleModelId ? modelsStore.props.modelSupportsAudio(singleModelId) : false
	);
</script>

<div class="space-y-6">
	<SettingsChatFields fields={fields.slice(0, splitAt)} {localConfig} {onConfigChange} />

	<div class="space-y-2" data-slot="voice-microphone">
		<Label class="text-label" for={MIC_KEY}>Microphone</Label>

		<Select.Root
			onValueChange={(value) => onConfigChange(MIC_KEY, value)}
			type="single"
			value={micCurrent}
		>
			<Select.Trigger class="w-full" id={MIC_KEY}>{micLabel}</Select.Trigger>

			<Select.Content>
				{#each micOptions as option (option.value)}
					<Select.Item label={option.label} value={option.value}>{option.label}</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>

		<p class="mt-1 text-caption text-muted-foreground">{SETTING_CONFIG_INFO[MIC_KEY]}</p>

		{#if micNamesHidden}
			<p class="text-caption text-muted-foreground">
				The browser names your microphones once you have allowed one, e.g. after your first
				recording.
			</p>
		{/if}
	</div>

	<div class="space-y-2" data-slot="voice-transcription-model">
		<Label class="text-label" for={KEY}>Transcription model</Label>

		{#if isRouter}
			<Select.Root
				onValueChange={(value) => onConfigChange(KEY, value)}
				type="single"
				value={current}
			>
				<Select.Trigger class="w-full" id={KEY}>{selectedLabel}</Select.Trigger>

				<Select.Content>
					<Select.Item
						label={VOICE_SETTINGS_LABEL.automatic}
						value={VOICE_TRANSCRIPTION_MODEL_AUTO}
					>
						{VOICE_SETTINGS_LABEL.automatic}
					</Select.Item>

					{#each optionGroups as group (group.heading)}
						<Select.Group>
							<Select.GroupHeading>{group.heading}</Select.GroupHeading>

							{#each group.items as option (option.value)}
								<Select.Item label={option.label} value={option.value}>{option.label}</Select.Item>
							{/each}
						</Select.Group>
					{/each}
				</Select.Content>
			</Select.Root>

			<p class="mt-1 text-caption text-muted-foreground">
				{SETTING_CONFIG_INFO[KEY]} Only models with audio input are listed.
			</p>

			<p class="text-caption text-muted-foreground">
				A model you pick here is loaded on the server when you dictate, alongside the chat model, if
				it isn't loaded already. A server that keeps one model at a time (<code>--models-max 1</code
				>) unloads the chat model to do it.
			</p>
		{:else}
			<p class="text-body-sm text-muted-foreground">
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
