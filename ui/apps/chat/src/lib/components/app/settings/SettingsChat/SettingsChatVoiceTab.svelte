<!--
	Voice section of the settings: the registry's fields, with the microphone and the transcription model between them.
	Those list this device's microphones and the server's models with audio input, which a static field cannot, so they
	are drawn here.
-->
<script lang="ts">
	import { NativeSelect } from '@ggml-org/llama-ui/app/forms';
	import { InsetList, InsetListRow } from '@ggml-org/llama-ui/app/sheets';
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import * as Select from '@ggml-org/llama-ui/ui/select';
	import {
		listMicrophones,
		type Microphone,
		VOICE_MICROPHONE_AUTO,
		VOICE_MICROPHONE_DEFAULT
	} from '@ggml-org/llama-voice';
	import {
		SettingsChatFields,
		SettingsChatFieldsInset
	} from '#lib/components/app/settings/index.ts';
	import {
		SETTING_CONFIG_INFO,
		SETTINGS_KEYS,
		VOICE_GGUF_EXTENSION_PATTERN,
		VOICE_SETTINGS_LABEL,
		VOICE_TRANSCRIPTION_MODEL_AUTO
	} from '#lib/constants/index.ts';
	import { modelsStore, serverStore } from '#lib/stores/index.ts';
	import { isTranscriptionOnlyModel } from '#lib/utils/voice-mode.ts';
	import { onMount, type Snippet } from 'svelte';

	interface Props {
		fields: SettingsFieldConfig[];
		/** Draw the section as a phone's inset list, for the settings sheet. */
		inset?: boolean;
		localConfig: SettingsConfigType;
		onConfigChange: (key: string, value: string | boolean) => void;
	}

	let { fields, inset = false, localConfig, onConfigChange }: Props = $props();

	/** Names the two pickers for the tests and demos that look for them. */
	const VOICE_TAB_SLOTS = {
		MICROPHONE: 'voice-microphone',
		TRANSCRIPTION_MODEL: 'voice-transcription-model'
	} as const;

	const VOICE_TAB_TEXT = {
		HAS_AUDIO: 'It has audio input, so it transcribes your voice.',
		MIC_NAMES_HIDDEN:
			'The browser names your microphones once you have allowed one, e.g. after your first recording.',
		MICROPHONE: 'Microphone',
		NO_AUDIO: 'It has no audio input, so voice input is not available on this server.',
		ONE_MODEL: 'This server runs one model, which transcribes your voice when it has audio input.',
		ONE_MODEL_NAMED: 'This server runs one model,',
		TRANSCRIPTION_MODEL: 'Transcription model',
		TRANSCRIPTION_NOTE:
			"Only models with audio input are listed. A model you pick here is loaded on the server when you dictate, if it isn't loaded already."
	} as const;

	const KEY = SETTINGS_KEYS.VOICE_TRANSCRIPTION_MODEL;

	// the pickers sit right after the Voice input field
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

{#snippet singleModelNote()}
	{#if singleModel}
		{VOICE_TAB_TEXT.ONE_MODEL_NAMED}
		<span class="font-medium text-foreground">{singleModel}</span>.
		{singleModelHasAudio ? VOICE_TAB_TEXT.HAS_AUDIO : VOICE_TAB_TEXT.NO_AUDIO}
	{:else}
		{VOICE_TAB_TEXT.ONE_MODEL}
	{/if}
{/snippet}

{#snippet microphonePicker()}
	<div class="space-y-2" data-slot={VOICE_TAB_SLOTS.MICROPHONE}>
		<Label class="text-label" for={MIC_KEY}>{VOICE_TAB_TEXT.MICROPHONE}</Label>

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
			<p class="text-caption text-muted-foreground">{VOICE_TAB_TEXT.MIC_NAMES_HIDDEN}</p>
		{/if}
	</div>
{/snippet}

{#snippet transcriptionModelPicker()}
	<div class="space-y-2" data-slot={VOICE_TAB_SLOTS.TRANSCRIPTION_MODEL}>
		<Label class="text-label" for={KEY}>{VOICE_TAB_TEXT.TRANSCRIPTION_MODEL}</Label>

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
			<p class="text-body-sm text-muted-foreground">{@render singleModelNote()}</p>
		{/if}
	</div>
{/snippet}

{#snippet insetPicker(
	label: string,
	value: string,
	valueLabel: string,
	pick: (value: string) => void,
	items: Array<{ label: string; value: string }>,
	itemGroups: Array<{ heading: string; items: Array<{ label: string; value: string }> }>,
	caption: Snippet,
	slot: string
)}
	<InsetList {caption} data-slot={slot}>
		<InsetListRow as="div" bodyClass="gap-4">
			<span class="flex-1 py-3">{label}</span>

			<NativeSelect
				displayLabel={valueLabel}
				groups={itemGroups.map((group) => ({ heading: group.heading, options: group.items }))}
				{label}
				onValueChange={pick}
				options={items}
				{value}
				variant="inset"
			/>
		</InsetListRow>
	</InsetList>
{/snippet}

{#snippet microphoneNote()}
	{SETTING_CONFIG_INFO[MIC_KEY]}

	{#if micNamesHidden}
		{VOICE_TAB_TEXT.MIC_NAMES_HIDDEN}
	{/if}
{/snippet}

{#snippet transcriptionNote()}
	{SETTING_CONFIG_INFO[KEY]}
	{VOICE_TAB_TEXT.TRANSCRIPTION_NOTE}
{/snippet}

{#if inset}
	<div class="flex flex-col gap-8">
		<SettingsChatFieldsInset fields={fields.slice(0, splitAt)} {localConfig} {onConfigChange} />

		{@render insetPicker(
			VOICE_TAB_TEXT.MICROPHONE,
			micCurrent,
			micLabel,
			(value) => onConfigChange(MIC_KEY, value),
			micOptions,
			[],
			microphoneNote,
			VOICE_TAB_SLOTS.MICROPHONE
		)}

		{#if isRouter}
			{@render insetPicker(
				VOICE_TAB_TEXT.TRANSCRIPTION_MODEL,
				current,
				selectedLabel,
				(value) => onConfigChange(KEY, value),
				[{ label: VOICE_SETTINGS_LABEL.automatic, value: VOICE_TRANSCRIPTION_MODEL_AUTO }],
				optionGroups,
				transcriptionNote,
				VOICE_TAB_SLOTS.TRANSCRIPTION_MODEL
			)}
		{:else}
			<p class="inset-list-caption" data-slot={VOICE_TAB_SLOTS.TRANSCRIPTION_MODEL}>
				{@render singleModelNote()}
			</p>
		{/if}

		<SettingsChatFieldsInset fields={fields.slice(splitAt)} {localConfig} {onConfigChange} />
	</div>
{:else}
	<div class="space-y-6">
		<SettingsChatFields fields={fields.slice(0, splitAt)} {localConfig} {onConfigChange} />

		{@render microphonePicker()}

		{@render transcriptionModelPicker()}

		<SettingsChatFields fields={fields.slice(splitAt)} {localConfig} {onConfigChange} />
	</div>
{/if}
