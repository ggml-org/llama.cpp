<script lang="ts" module>
	import { Input } from '@ggml-org/llama-ui/ui/input';
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		component: Label,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Components/Forms/Label'
	});
</script>

<Story asChild name="Default">
	<div class="grid w-72 gap-2">
		<Label for="story-api-key">API key</Label>

		<Input id="story-api-key" placeholder="sk-..." />
	</div>
</Story>
