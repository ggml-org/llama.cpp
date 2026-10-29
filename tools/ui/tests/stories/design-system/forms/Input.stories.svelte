<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Input } from '$lib/components/ui/input';

	const { Story } = defineMeta({
		component: Input,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Forms/Input'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2}>
		<StateGalleryItem label="Empty">
			<Input placeholder="http://localhost:8080" />
		</StateGalleryItem>

		<StateGalleryItem label="Filled">
			<Input value="http://localhost:8080" />
		</StateGalleryItem>

		<StateGalleryItem label="Invalid">
			<Input aria-invalid="true" value="not a url" />
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<Input disabled value="Read only" />
		</StateGalleryItem>

		<StateGalleryItem label="File">
			<Input type="file" />
		</StateGalleryItem>
	</StateGallery>
</Story>
