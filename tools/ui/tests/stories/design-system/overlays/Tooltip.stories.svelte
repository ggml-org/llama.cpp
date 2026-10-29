<script lang="ts" module>
	import { Copy } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';

	const { Story } = defineMeta({
		component: Tooltip.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Overlays/Tooltip'
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
