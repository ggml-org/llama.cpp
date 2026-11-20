<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { TruncatedText } from '#lib/components/app/index.ts';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	// one line of text that shows the full value in a tooltip only when it is cut off
	const { Story } = defineMeta({
		component: TruncatedText,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Misc/TruncatedText'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2}>
		<StateGalleryItem label="Fits">
			<div class="w-56 text-body-sm">
				<TruncatedText text="qwen3-8b" />
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Truncated" note="hover for the tooltip">
			<div class="w-56 text-body-sm">
				<TruncatedText text="unsloth/Meta-Llama-3.1-70B-Instruct-GGUF:Q4_K_M" />
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>
