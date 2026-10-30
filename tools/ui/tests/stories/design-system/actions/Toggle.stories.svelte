<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { Bold, Italic, WrapText } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Button } from '$lib/components/ui/button';
	import { Toggle } from '$lib/components/ui/toggle';

	const { Story } = defineMeta({
		component: Toggle,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Actions/Toggle'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={4} minItemWidth="10rem">
		<StateGalleryItem label="Off">
			<Toggle aria-label="Bold" size="icon"><Bold /></Toggle>
		</StateGalleryItem>

		<StateGalleryItem label="On">
			<Toggle aria-label="Bold" pressed size="icon"><Bold /></Toggle>
		</StateGalleryItem>

		<StateGalleryItem label="Outline">
			<div class="flex gap-2">
				<Toggle aria-label="Italic" size="icon" variant="outline"><Italic /></Toggle>

				<Toggle aria-label="Italic" pressed size="icon" variant="outline"><Italic /></Toggle>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Sizes" note="sm, default, lg">
			<div class="flex items-center gap-2">
				<Toggle size="sm" variant="outline">Wrap</Toggle>

				<Toggle variant="outline">Wrap</Toggle>

				<Toggle size="lg" variant="outline">Wrap</Toggle>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<Toggle aria-label="Bold" disabled size="icon"><Bold /></Toggle>
		</StateGalleryItem>

		<StateGalleryItem label="Next to Button" note="same height, padding and corners" span={3}>
			<div class="flex items-center gap-2">
				<Button variant="outline">Button</Button>

				<Toggle variant="outline">Toggle</Toggle>

				<Button variant="outline"><WrapText /> Wrap</Button>

				<Toggle pressed variant="outline"><WrapText /> Wrap</Toggle>

				<Button aria-label="Bold" size="icon" variant="outline"><Bold /></Button>

				<Toggle aria-label="Bold" size="icon" variant="outline"><Bold /></Toggle>
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>
