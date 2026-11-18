<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the safe-area utilities from src/lib/styles/safe-area.css, in phone frames that simulate an
	// installed iPhone app's insets (they are 0 in a desktop browser)
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Foundations/Safe areas'
	});

	/** An iPhone's status bar and home indicator, as an installed app sees them. */
	const SIMULATED_INSETS = '--safe-area-top: 47px; --safe-area-bottom: 34px;';
</script>

{#snippet phone(content: () => ReturnType<import('svelte').Snippet>)}
	<div
		class="relative h-[34rem] w-64 overflow-hidden rounded-[2.5rem] border-4 border-foreground/80 bg-background"
		style={SIMULATED_INSETS}
	>
		{@render content()}

		<!-- the insets, tinted over whatever is drawn under them -->
		<div
			class="pointer-events-none absolute inset-x-0 top-0 flex h-(--safe-area-top) items-center justify-center bg-rose-500/20 text-[10px] font-medium text-rose-700 dark:text-rose-300"
		>
			status bar · --safe-area-top
		</div>

		<div
			class="pointer-events-none absolute inset-x-0 bottom-0 flex h-(--safe-area-bottom) items-center justify-center bg-rose-500/20 text-[10px] font-medium text-rose-700 dark:text-rose-300"
		>
			home indicator · --safe-area-bottom
		</div>
	</div>
{/snippet}

{#snippet unpadded()}
	<div class="flex h-full flex-col justify-between p-4 text-sm">
		<p class="font-medium">Page title under the status bar</p>

		<p class="text-muted-foreground">Bottom bar over the home indicator</p>
	</div>
{/snippet}

{#snippet page()}
	<div class="flex h-full flex-col justify-between px-4 pt-safe-12 pb-safe-2 text-sm">
		<p class="font-medium">Page title, 3rem below the status bar</p>

		<p class="rounded-full bg-muted px-3 py-2 text-center text-muted-foreground">
			Bottom bar, clear of the home indicator
		</p>
	</div>
{/snippet}

{#snippet sheet()}
	<div class="relative h-full bg-muted/50 px-4 pt-safe text-sm">
		<p class="pt-2 text-muted-foreground">Type a message...</p>

		<!-- absolute positions ignore the sheet's padding: the control takes its own offset -->
		<span
			class="absolute top-safe-2.5 right-2.5 flex size-8 items-center justify-center rounded-full bg-background text-xs shadow-sm"
		>
			⤡
		</span>
	</div>
{/snippet}

<Story asChild name="Default">
	<StateGallery columns={3} minItemWidth="16rem">
		<StateGalleryItem label="Unpadded" note="what to avoid">
			{@render phone(unpadded)}
		</StateGalleryItem>

		<StateGalleryItem label="pt-safe-12 · pb-safe-2" note="a page and its bottom bar">
			{@render phone(page)}
		</StateGalleryItem>

		<StateGalleryItem label="pt-safe · top-safe-2.5" note="a sheet that fills the screen">
			{@render phone(sheet)}
		</StateGalleryItem>
	</StateGallery>
</Story>
