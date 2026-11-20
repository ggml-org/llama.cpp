<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { SearchInput } from '@ggml-org/llama-ui/app/forms';

	// the search field of the sidebar, model lists and pickers, and how matches are marked
	const { Story } = defineMeta({
		component: SearchInput,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/App components/SearchInput'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2}>
		<StateGalleryItem label="Empty">
			<SearchInput placeholder="Search conversations..." />
		</StateGalleryItem>

		<StateGalleryItem label="With value" note="clear button">
			<SearchInput value="llama" />
		</StateGalleryItem>

		<StateGalleryItem label="Small">
			<SearchInput placeholder="Filter models..." size="sm" />
		</StateGalleryItem>
	</StateGallery>
</Story>
