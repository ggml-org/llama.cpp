<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Label } from '$lib/components/ui/label';

	const { Story } = defineMeta({
		component: Checkbox,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Forms/Checkbox'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={5} minItemWidth="8rem">
		<StateGalleryItem label="Off">
			<Label class="flex items-center gap-2"><Checkbox /> Tools</Label>
		</StateGalleryItem>

		<StateGalleryItem label="On">
			<Label class="flex items-center gap-2"><Checkbox checked /> Tools</Label>
		</StateGalleryItem>

		<StateGalleryItem label="Indeterminate">
			<Label class="flex items-center gap-2"><Checkbox indeterminate /> Tools</Label>
		</StateGalleryItem>

		<StateGalleryItem label="Invalid">
			<Label class="flex items-center gap-2"><Checkbox aria-invalid="true" /> Accept</Label>
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<Label class="flex items-center gap-2"><Checkbox disabled /> Tools</Label>
		</StateGalleryItem>
	</StateGallery>
</Story>
