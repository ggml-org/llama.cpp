<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { HighlightedMatch, SearchInput } from '$lib/components/app';

	// the search field of the sidebar, model lists and pickers, and how matches are marked
	const { Story } = defineMeta({
		component: SearchInput,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Patterns/SearchInput'
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

		<StateGalleryItem label="HighlightedMatch">
			<span class="text-body-sm">
				<HighlightedMatch query="llama" text="Meta-Llama-3.1-8B-Instruct" />
			</span>
		</StateGalleryItem>
	</StateGallery>
</Story>
