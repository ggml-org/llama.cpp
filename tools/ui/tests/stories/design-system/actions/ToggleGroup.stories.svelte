<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { AlignCenter, AlignLeft, AlignRight } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';

	const { Story } = defineMeta({
		component: ToggleGroup.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Actions/ToggleGroup'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2} minItemWidth="16rem">
		<StateGalleryItem label="Single">
			<ToggleGroup.Root type="single" value="left">
				<ToggleGroup.Item aria-label="Align left" value="left"><AlignLeft /></ToggleGroup.Item>

				<ToggleGroup.Item aria-label="Align center" value="center"><AlignCenter /></ToggleGroup.Item
				>

				<ToggleGroup.Item aria-label="Align right" value="right"><AlignRight /></ToggleGroup.Item>
			</ToggleGroup.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Multiple" note="outline">
			<ToggleGroup.Root type="multiple" value={['tools', 'vision']} variant="outline">
				<ToggleGroup.Item value="tools">Tools</ToggleGroup.Item>

				<ToggleGroup.Item value="vision">Vision</ToggleGroup.Item>

				<ToggleGroup.Item value="audio">Audio</ToggleGroup.Item>
			</ToggleGroup.Root>
		</StateGalleryItem>
	</StateGallery>
</Story>
