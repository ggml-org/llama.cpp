<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Separator } from '$lib/components/ui/separator';

	const { Story } = defineMeta({
		component: Separator,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Display/Separator'
	});
</script>

<Story asChild name="Default">
	<div class="w-80 text-sm">
		<p class="font-medium">Qwen3.8-27B</p>

		<p class="text-muted-foreground">Loaded, 131k context</p>

		<Separator class="my-4" />

		<div class="flex h-5 items-center gap-4">
			<span>Chat</span>

			<Separator orientation="vertical" />

			<span>Models</span>

			<Separator orientation="vertical" />

			<span>Settings</span>
		</div>
	</div>
</Story>
