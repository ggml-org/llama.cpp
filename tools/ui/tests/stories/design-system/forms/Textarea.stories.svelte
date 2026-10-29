<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Textarea } from '$lib/components/ui/textarea';

	const { Story } = defineMeta({
		component: Textarea,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Forms/Textarea'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2}>
		<StateGalleryItem label="Empty">
			<Textarea placeholder="You are a helpful assistant." />
		</StateGalleryItem>

		<StateGalleryItem label="Filled">
			<Textarea value="You are a concise assistant. Answer in plain English and cite sources." />
		</StateGalleryItem>

		<StateGalleryItem label="Invalid">
			<Textarea aria-invalid="true" value="{'{'} not valid json" />
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<Textarea disabled value="Set by the server" />
		</StateGalleryItem>
	</StateGallery>
</Story>
