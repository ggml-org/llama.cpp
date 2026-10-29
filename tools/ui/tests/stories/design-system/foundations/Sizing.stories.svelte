<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { Settings } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// control heights and icon sizes from src/lib/styles/tokens/semantic.css, and the 4px spacing grid
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Foundations/Sizing'
	});

	const heights = [
		{ name: 'h-control-sm', note: '32px' },
		{ name: 'h-control-md', note: '36px, default' },
		{ name: 'h-control-lg', note: '40px' }
	];
	const icons = [
		{ name: 'size-icon-xs', note: '12px' },
		{ name: 'size-icon-sm', note: '14px' },
		{ name: 'size-icon-md', note: '16px, default' },
		{ name: 'size-icon-lg', note: '20px' }
	];
	const spacing = [
		{ name: 'xs', px: 4 },
		{ name: 'sm', px: 8 },
		{ name: 'md', px: 12 },
		{ name: 'lg', px: 16 },
		{ name: 'xl', px: 24 },
		{ name: '2xl', px: 32 }
	];
</script>

<Story asChild name="Default">
	<StateGallery columns={1}>
		<StateGalleryItem label="Control heights">
			<div class="flex items-end gap-3">
				{#each heights as height (height.name)}
					<div class="flex flex-col items-center gap-2">
						<div
							class="flex w-20 items-center justify-center rounded-control border border-line-default bg-surface-muted {height.name}"
						></div>

						<span class="font-mono text-[11px] text-fg-muted">{height.name}</span>

						<span class="text-caption text-fg-muted">{height.note}</span>
					</div>
				{/each}
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Icon sizes">
			<div class="flex items-end gap-5">
				{#each icons as icon (icon.name)}
					<div class="flex flex-col items-center gap-2">
						<Settings class={icon.name} />

						<span class="font-mono text-[11px] text-fg-muted">{icon.name}</span>

						<span class="text-caption text-fg-muted">{icon.note}</span>
					</div>
				{/each}
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Spacing" note="4px grid">
			<div class="flex flex-col gap-2">
				{#each spacing as step (step.name)}
					<div class="flex items-center gap-3">
						<span class="w-8 font-mono text-[11px] text-fg-muted">{step.name}</span>

						<div style:width="{step.px}px" class="h-3 rounded-sm bg-muted-foreground/40"></div>

						<span class="text-caption text-fg-muted">{step.px}px</span>
					</div>
				{/each}
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>
