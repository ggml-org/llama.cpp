<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import * as RadioGroup from '@ggml-org/llama-ui/ui/radio-group';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		component: RadioGroup.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Components/Forms/RadioGroup'
	});

	const options = ['System', 'Light', 'Dark'];
</script>

<Story asChild name="Default">
	<StateGallery columns={2} minItemWidth="12rem">
		<StateGalleryItem label="Enabled">
			<RadioGroup.Root value="System">
				{#each options as option (option)}
					<Label class="flex items-center gap-2">
						<RadioGroup.Item value={option} />
						{option}
					</Label>
				{/each}
			</RadioGroup.Root>
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<RadioGroup.Root disabled value="Dark">
				{#each options as option (option)}
					<Label class="flex items-center gap-2">
						<RadioGroup.Item value={option} />
						{option}
					</Label>
				{/each}
			</RadioGroup.Root>
		</StateGalleryItem>
	</StateGallery>
</Story>
