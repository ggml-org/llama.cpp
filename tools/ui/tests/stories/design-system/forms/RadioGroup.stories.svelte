<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Label } from '$lib/components/ui/label';
	import * as RadioGroup from '$lib/components/ui/radio-group';

	const { Story } = defineMeta({
		component: RadioGroup.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Forms/RadioGroup'
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
