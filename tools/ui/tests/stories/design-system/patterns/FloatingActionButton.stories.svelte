<script lang="ts" module>
	import { Plus } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { FloatingActionButton } from '$lib/components/app';

	// a page's main action on a phone (New project, New prompt): a round key in the
	// bottom-right corner; from md up the page keeps it as a button in its header
	const { Story } = defineMeta({
		component: FloatingActionButton,
		globals: { viewport: { isRotated: false, value: 'mobile2' } },
		parameters: { layout: 'fullscreen' },
		title: 'Design system/Patterns/FloatingActionButton'
	});
</script>

<Story asChild name="On a page">
	<div class="flex min-h-dvh flex-col gap-6 px-4 pt-12 pb-safe-24">
		<div class="flex flex-col gap-1">
			<h1 class="text-2xl font-semibold tracking-tight">Projects</h1>

			<p class="text-sm text-muted-foreground">
				The header's New project button gives way to the floating key on a phone.
			</p>
		</div>

		{#each [1, 2, 3, 4, 5] as card (card)}
			<div class="h-28 rounded-xl border border-border/30 bg-muted/40"></div>
		{/each}
	</div>

	<FloatingActionButton icon={Plus} label="New project" />
</Story>
