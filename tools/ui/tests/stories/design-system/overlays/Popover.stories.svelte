<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import * as Popover from '$lib/components/ui/popover';

	const { Story } = defineMeta({
		component: Popover.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Overlays/Popover'
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
					<p class="text-sm font-medium">Sampling</p>

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
