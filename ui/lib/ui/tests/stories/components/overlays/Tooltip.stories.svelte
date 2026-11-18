<script lang="ts" module>
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import { Copy } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		component: Tooltip.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Components/Overlays/Tooltip'
	});
</script>

<Story asChild name="Default">
	<div class="flex h-32 items-end">
		<Tooltip.Root open>
			<Tooltip.Trigger>
				{#snippet child({ props })}
					<Button {...props} aria-label="Copy" size="icon" variant="outline"><Copy /></Button>
				{/snippet}
			</Tooltip.Trigger>

			<Tooltip.Content>Copy to clipboard</Tooltip.Content>
		</Tooltip.Root>
	</div>
</Story>
