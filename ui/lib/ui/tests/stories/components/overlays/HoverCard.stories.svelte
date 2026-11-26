<script lang="ts" module>
	import * as HoverCard from '@ggml-org/llama-ui/ui/hover-card';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		component: HoverCard.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Components/Overlays/HoverCard'
	});
</script>

<Story asChild name="Default">
	<div class="h-56">
		<HoverCard.Root open>
			<HoverCard.Trigger>
				{#snippet child({ props })}
					<a {...props} class="text-body-sm underline-offset-4 hover:underline" href="#hover-card"
						>ggml-org/Qwen3.8-27B-GGUF</a
					>
				{/snippet}
			</HoverCard.Trigger>

			<HoverCard.Content class="w-72">
				<div class="space-y-1">
					<p class="text-label">Qwen3.8-27B</p>

					<p class="text-body-sm text-muted-foreground">
						131k context, tool use and vision. Quantized to Q4_K_M, 16.4 GB.
					</p>
				</div>
			</HoverCard.Content>
		</HoverCard.Root>
	</div>
</Story>
