<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';

	const { Story } = defineMeta({
		component: Label,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Forms/Label'
	});
</script>

<Story asChild name="Default">
	<div class="grid w-72 gap-2">
		<Label for="story-api-key">API key</Label>

		<Input id="story-api-key" placeholder="sk-..." />
	</div>
</Story>
