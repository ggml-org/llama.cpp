<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { InputWithSuggestions } from '$lib/components/app';

	// a field with an autocomplete list under it; the caller owns the suggestions and the keys
	const { Story } = defineMeta({
		component: InputWithSuggestions,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Patterns/InputWithSuggestions'
	});

	const noop = () => {};
</script>

<Story asChild name="Default">
	<StateGallery columns={3} minItemWidth="16rem">
		<StateGalleryItem label="Idle">
			<InputWithSuggestions
				name="model"
				onBlur={noop}
				onFocus={noop}
				onInput={noop}
				onKeydown={noop}
				onSelectSuggestion={noop}
				value=""
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Suggestions">
			<div class="h-48">
				<InputWithSuggestions
					autocompleteIndex={1}
					isAutocompleteActive
					name="model"
					onBlur={noop}
					onFocus={noop}
					onInput={noop}
					onKeydown={noop}
					onSelectSuggestion={noop}
					suggestions={[
						'ggml-org/gemma-3-4b-it-GGUF',
						'ggml-org/Qwen3-8B-GGUF',
						'ggml-org/gpt-oss-20b-GGUF'
					]}
					value="ggml-org/"
				/>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Loading">
			<InputWithSuggestions
				isAutocompleteActive
				isLoadingSuggestions
				name="model"
				onBlur={noop}
				onFocus={noop}
				onInput={noop}
				onKeydown={noop}
				onSelectSuggestion={noop}
				value="ggml-org/"
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
