<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Label } from '$lib/components/ui/label';
	import { Switch } from '$lib/components/ui/switch';

	const { Story } = defineMeta({
		component: Switch,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Forms/Switch'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={4} minItemWidth="9rem">
		<StateGalleryItem label="Off">
			<Label class="flex items-center gap-2"><Switch /> Stream</Label>
		</StateGalleryItem>

		<StateGalleryItem label="On">
			<Label class="flex items-center gap-2"><Switch checked /> Stream</Label>
		</StateGalleryItem>

		<StateGalleryItem label="Disabled, off">
			<Label class="flex items-center gap-2"><Switch disabled /> Stream</Label>
		</StateGalleryItem>

		<StateGalleryItem label="Disabled, on">
			<Label class="flex items-center gap-2"><Switch checked disabled /> Stream</Label>
		</StateGalleryItem>
	</StateGallery>
</Story>
