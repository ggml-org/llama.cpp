<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { Bold, Italic } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
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
			<Toggle aria-label="Bold"><Bold /></Toggle>
		</StateGalleryItem>

		<StateGalleryItem label="On">
			<Toggle aria-label="Bold" pressed><Bold /></Toggle>
		</StateGalleryItem>

		<StateGalleryItem label="Outline">
			<div class="flex gap-2">
				<Toggle aria-label="Italic" variant="outline"><Italic /></Toggle>

				<Toggle aria-label="Italic" pressed variant="outline"><Italic /></Toggle>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Sizes" note="sm, default, lg">
			<div class="flex items-center gap-2">
				<Toggle size="sm">Wrap</Toggle>

				<Toggle>Wrap</Toggle>

				<Toggle size="lg">Wrap</Toggle>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<Toggle aria-label="Bold" disabled><Bold /></Toggle>
		</StateGalleryItem>
	</StateGallery>
</Story>
