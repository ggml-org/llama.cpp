<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Switch } from '$lib/components/ui/switch';
	import { Textarea } from '$lib/components/ui/textarea';

	// the form fields share one story, since they are reviewed together
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Fields'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2}>
		<StateGalleryItem label="Input">
			<Input placeholder="http://localhost:8080" />
		</StateGalleryItem>

		<StateGalleryItem label="Input" note="disabled">
			<Input disabled value="Read only" />
		</StateGalleryItem>

		<StateGalleryItem label="Input" note="invalid">
			<Input aria-invalid="true" value="not a url" />
		</StateGalleryItem>

		<StateGalleryItem label="Textarea">
			<Textarea placeholder="You are a helpful assistant." />
		</StateGalleryItem>

		<StateGalleryItem label="Switch">
			<div class="flex items-center gap-4">
				<Switch />

				<Switch checked />

				<Switch disabled />
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Checkbox">
			<div class="flex items-center gap-4">
				<Label class="flex items-center gap-2"><Checkbox /> Off</Label>

				<Label class="flex items-center gap-2"><Checkbox checked /> On</Label>

				<Label class="flex items-center gap-2"><Checkbox disabled /> Disabled</Label>
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>
