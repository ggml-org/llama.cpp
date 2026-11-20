<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Collapsible from '@ggml-org/llama-ui/ui/collapsible';
	import { ChevronsUpDown } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		component: Collapsible.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Components/Display/Collapsible'
	});
</script>

{#snippet section(open: boolean)}
	<Collapsible.Root class="w-72 space-y-2" {open}>
		<div class="flex items-center justify-between">
			<span class="text-label">Advanced sampling</span>

			<Collapsible.Trigger>
				{#snippet child({ props })}
					<Button {...props} aria-label="Toggle" size="icon" variant="ghost">
						<ChevronsUpDown />
					</Button>
				{/snippet}
			</Collapsible.Trigger>
		</div>

		<div class="rounded-md border px-4 py-2 font-mono text-body-sm">temperature 0.8</div>

		<Collapsible.Content class="space-y-2">
			<div class="rounded-md border px-4 py-2 font-mono text-body-sm">top_p 0.95</div>

			<div class="rounded-md border px-4 py-2 font-mono text-body-sm">min_p 0.05</div>
		</Collapsible.Content>
	</Collapsible.Root>
{/snippet}

<Story asChild name="Default">
	<StateGallery columns={2} minItemWidth="18rem">
		<StateGalleryItem label="Closed">
			{@render section(false)}
		</StateGalleryItem>

		<StateGalleryItem label="Open">
			{@render section(true)}
		</StateGalleryItem>
	</StateGallery>
</Story>
