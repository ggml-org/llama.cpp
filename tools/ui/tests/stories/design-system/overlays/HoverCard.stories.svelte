<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Button } from '$lib/components/ui/button';
	import * as HoverCard from '$lib/components/ui/hover-card';

	const { Story } = defineMeta({
		component: HoverCard.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Overlays/HoverCard'
	});
</script>

<Story asChild name="Default">
	<div class="h-56">
		<HoverCard.Root open>
			<HoverCard.Trigger>
				{#snippet child({ props })}
					<Button {...props} variant="link">ggml-org/Qwen3.8-27B-GGUF</Button>
				{/snippet}
			</HoverCard.Trigger>

			<HoverCard.Content class="w-72">
				<div class="space-y-1">
					<p class="text-sm font-medium">Qwen3.8-27B</p>

					<p class="text-sm text-muted-foreground">
						131k context, tool use and vision. Quantized to Q4_K_M, 16.4 GB.
					</p>
				</div>
			</HoverCard.Content>
		</HoverCard.Root>
	</div>
</Story>
