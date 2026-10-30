<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { AlignCenter, AlignLeft, AlignRight } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import type { ToggleSize } from '$lib/components/ui/toggle';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';

	const { Story } = defineMeta({
		component: ToggleGroup.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Actions/ToggleGroup'
	});

	const sizes: NonNullable<ToggleSize>[] = ['icon-sm', 'icon', 'icon-lg'];
</script>

<Story asChild name="Default">
	<StateGallery columns={2} minItemWidth="16rem">
		<StateGalleryItem label="Single" note="icons">
			<ToggleGroup.Root size="icon-sm" type="single" value="center">
				<ToggleGroup.Item aria-label="Align left" value="left"><AlignLeft /></ToggleGroup.Item>

				<ToggleGroup.Item aria-label="Align center" value="center"><AlignCenter /></ToggleGroup.Item
				>

				<ToggleGroup.Item aria-label="Align right" value="right"><AlignRight /></ToggleGroup.Item>
			</ToggleGroup.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Nothing pressed">
			<ToggleGroup.Root type="multiple" value={[]}>
				<ToggleGroup.Item value="tools">Tools</ToggleGroup.Item>

				<ToggleGroup.Item value="vision">Vision</ToggleGroup.Item>

				<ToggleGroup.Item value="audio">Audio</ToggleGroup.Item>
			</ToggleGroup.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Multiple" note="the unpressed keys mute">
			<ToggleGroup.Root type="multiple" value={['tools', 'vision']}>
				<ToggleGroup.Item value="tools">Tools</ToggleGroup.Item>

				<ToggleGroup.Item value="vision">Vision</ToggleGroup.Item>

				<ToggleGroup.Item value="audio">Audio</ToggleGroup.Item>
			</ToggleGroup.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Sizes" note="icon-sm, icon, icon-lg">
			<div class="flex items-center gap-3">
				{#each sizes as size (size)}
					<ToggleGroup.Root {size} type="single" value="left">
						<ToggleGroup.Item aria-label="Align left" value="left"><AlignLeft /></ToggleGroup.Item>

						<ToggleGroup.Item aria-label="Align right" value="right"
							><AlignRight /></ToggleGroup.Item
						>
					</ToggleGroup.Root>
				{/each}
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<ToggleGroup.Root disabled size="icon-sm" type="single" value="left">
				<ToggleGroup.Item aria-label="Align left" value="left"><AlignLeft /></ToggleGroup.Item>

				<ToggleGroup.Item aria-label="Align right" value="right"><AlignRight /></ToggleGroup.Item>
			</ToggleGroup.Root>
		</StateGalleryItem>
	</StateGallery>
</Story>
