<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { NativeSelect } from '@ggml-org/llama-ui/app/forms';

	// a select that keeps the library's trigger and opens the OS's own picker: a wheel or a sheet
	// on a phone, a native menu on a desktop
	const { Story } = defineMeta({
		component: NativeSelect,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/App components/NativeSelect'
	});

	const LANGUAGES = [
		{ label: 'English', value: 'en' },
		{ label: 'Polish', value: 'pl' },
		{ label: 'Japanese', value: 'ja' }
	];

	const MODEL_GROUPS = [
		{
			heading: 'Speech to text',
			options: [{ label: 'whisper-large-v3', value: 'whisper' }]
		},
		{
			heading: 'Chat models with audio input',
			options: [{ label: 'gemma-3n-E4B', value: 'gemma' }]
		}
	];
</script>

<script lang="ts">
	let language = $state('en');
	let model = $state('auto');
</script>

<Story asChild name="Default">
	<StateGallery columns={2}>
		<StateGalleryItem label="Default" note="the boxed trigger">
			<div class="w-64">
				<NativeSelect
					label="Language"
					onValueChange={(value) => (language = value)}
					options={LANGUAGES}
					value={language}
				/>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Inset" note="a value at the end of a list row">
			<div class="flex w-64 justify-end">
				<NativeSelect
					label="Language"
					onValueChange={(value) => (language = value)}
					options={LANGUAGES}
					value={language}
					variant="inset"
				/>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Grouped options">
			<div class="w-64">
				<NativeSelect
					groups={MODEL_GROUPS}
					label="Transcription model"
					onValueChange={(value) => (model = value)}
					options={[{ label: 'Automatic', value: 'auto' }]}
					value={model}
				/>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Nothing picked" note="the placeholder">
			<div class="w-64">
				<NativeSelect label="Language" onValueChange={() => {}} options={LANGUAGES} value="" />
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<div class="w-64">
				<NativeSelect
					disabled
					label="Language"
					onValueChange={() => {}}
					options={LANGUAGES}
					value="en"
				/>
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>
