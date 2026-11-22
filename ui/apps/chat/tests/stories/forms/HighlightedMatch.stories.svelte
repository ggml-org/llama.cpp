<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { HighlightedMatch } from '#lib/components/app/index.ts';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	// how a search query is marked in the sidebar's, the model lists' and the pickers' results
	const { Story } = defineMeta({
		component: HighlightedMatch,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Forms/HighlightedMatch'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2}>
		<StateGalleryItem label="Match" note="the query marked in the result">
			<span class="text-body-sm">
				<HighlightedMatch query="llama" text="Meta-Llama-3.1-8B-Instruct" />
			</span>
		</StateGalleryItem>

		<StateGalleryItem label="No match">
			<span class="text-body-sm">
				<HighlightedMatch query="qwen" text="Meta-Llama-3.1-8B-Instruct" />
			</span>
		</StateGalleryItem>
	</StateGallery>
</Story>
