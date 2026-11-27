<script lang="ts" module>
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { Input } from '@ggml-org/llama-ui/ui/input';
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import * as Popover from '@ggml-org/llama-ui/ui/popover';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		component: Popover.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Components/Overlays/Popover'
	});
</script>

<Story asChild name="Default">
	<div class="h-72">
		<Popover.Root open>
			<Popover.Trigger>
				{#snippet child({ props })}
					<Button {...props} variant="outline">Sampling</Button>
				{/snippet}
			</Popover.Trigger>

			<Popover.Content class="w-72">
				<div class="grid gap-3">
					<p class="text-label">Sampling</p>

					<div class="grid grid-cols-3 items-center gap-2">
						<Label for="story-temp">Temperature</Label>

						<Input class="col-span-2 h-8" id="story-temp" value="0.8" />
					</div>

					<div class="grid grid-cols-3 items-center gap-2">
						<Label for="story-top-p">Top P</Label>

						<Input class="col-span-2 h-8" id="story-top-p" value="0.95" />
					</div>
				</div>
			</Popover.Content>
		</Popover.Root>
	</div>
</Story>
