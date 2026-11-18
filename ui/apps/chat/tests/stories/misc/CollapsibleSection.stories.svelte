<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { CollapsibleSection } from '#lib/components/app/index.ts';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	// the animated show and hide section used by sidebar groups and model lists
	const { Story } = defineMeta({
		component: CollapsibleSection,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Misc/CollapsibleSection'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2} minItemWidth="16rem">
		{#each [true, false] as defaultOpen (defaultOpen)}
			<StateGalleryItem label={defaultOpen ? 'Open' : 'Closed'}>
				<CollapsibleSection {defaultOpen}>
					{#snippet trigger()}
						<span class="text-label">Pinned</span>
					{/snippet}

					<ul class="space-y-1 pt-2 text-body-sm text-fg-muted">
						<li>Explain quantization</li>

						<li>Rust borrow checker notes</li>

						<li>Trip plan for Lisbon</li>
					</ul>
				</CollapsibleSection>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>
