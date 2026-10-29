<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { CircleAlert, Info } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import * as Alert from '$lib/components/ui/alert';

	const { Story } = defineMeta({
		component: Alert.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Display/Alert'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2} minItemWidth="20rem">
		<StateGalleryItem label="Default">
			<Alert.Root>
				<Info />

				<Alert.Title>Model is loading</Alert.Title>

				<Alert.Description>The first reply can take a few seconds.</Alert.Description>
			</Alert.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Destructive">
			<Alert.Root variant="destructive">
				<CircleAlert />

				<Alert.Title>Context is full</Alert.Title>

				<Alert.Description>Start a new chat or remove some attachments.</Alert.Description>
			</Alert.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Title only">
			<Alert.Root>
				<Info />

				<Alert.Title>Saved to this browser</Alert.Title>
			</Alert.Root>
		</StateGalleryItem>
	</StateGallery>
</Story>
