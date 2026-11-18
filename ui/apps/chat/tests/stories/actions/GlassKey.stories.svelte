<script lang="ts" module>
	import { ArrowDown } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { GlassKey, MenuIcon } from '#lib/components/app/index.ts';
	import { ICON_CLASS_DEFAULT } from '#lib/constants/index.ts';

	// a round matte glass key over the page: a phone's menu and context gauge keys, and the key
	// back to a chat's bottom; the text under them shows how the glass softens what scrolls by
	const { Story } = defineMeta({
		component: GlassKey,
		parameters: { layout: 'fullscreen' },
		title: 'Llama Chat/Actions/GlassKey'
	});

	const LINES = Array.from(
		{ length: 8 },
		(_, index) => `Line ${index + 1}: what scrolls under a glass key softens behind its icon.`
	);
</script>

<Story asChild name="Over the page">
	<div class="relative min-h-80 p-6">
		<div class="flex flex-col gap-2 text-sm">
			{#each LINES as line (line)}
				<p>{line}</p>
			{/each}
		</div>

		<div class="absolute inset-x-6 top-10 flex items-center justify-between">
			<GlassKey icon={MenuIcon} label="Expand navigation" />

			<GlassKey
				icon={ArrowDown}
				iconClass={ICON_CLASS_DEFAULT}
				label="Scroll to bottom"
				size="sm"
			/>
		</div>
	</div>
</Story>
