<script lang="ts">
	import { Settings } from '@lucide/svelte';
	import { ScrollCarousel } from '$lib/components/app';
	import { ICON_CLASS_DEFAULT, UI_DATA_ATTRS } from '$lib/constants';
	import { BooleanString } from '$lib/enums';
	import { useScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';
	import type { SettingsSection, SettingsSectionTitle } from '$lib/types';
	import { onMount, tick } from 'svelte';

	interface Props {
		sections: SettingsSection[];
		isActive: (section: SettingsSection) => boolean;
		getHref?: (section: SettingsSection) => string;
		onSectionChange?: (section: SettingsSectionTitle) => void;
	}

	let { getHref, isActive, onSectionChange, sections }: Props = $props();

	const carousel = useScrollCarousel();

	onMount(async () => {
		await tick();

		if (carousel.scrollContainer) {
			const activeTab = carousel.scrollContainer.querySelector(
				`[${UI_DATA_ATTRS.ACTIVE}="${BooleanString.TRUE}"]`
			);

			if (activeTab instanceof HTMLElement) {
				carousel.scrollToCenter(activeTab);
			}
		}
	});

	export function updateCarousel() {
		setTimeout(carousel.updateScrollButtons, 100);
	}
</script>

<div class="sticky top-0 z-10 flex flex-col bg-background md:hidden">
	<div class="flex items-center gap-2 px-4 pt-4 pb-2 md:pt-6">
		<Settings class="h-5 w-5 md:h-6 md:w-6" />

		<h1 class="text-xl font-semibold md:text-2xl">Settings</h1>
	</div>

	<div class="border-b border-border/30 py-2">
		<ScrollCarousel {carousel} alwaysShowArrows containerClass="py-2" innerClass="gap-2">
			{#each sections as section (section.title)}
				{#if getHref}
					<a
						class="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm whitespace-nowrap no-underline transition-colors first:ml-4 last:mr-4 hover:bg-accent {isActive(
							section
						)
							? 'bg-accent text-accent-foreground'
							: 'text-muted-foreground'}"
						{...{ [UI_DATA_ATTRS.ACTIVE]: isActive(section) }}
						href={getHref(section)}
						onclick={(e: MouseEvent) => {
							carousel.scrollToCenter(e.currentTarget as HTMLElement);
						}}
					>
						<section.icon class="{ICON_CLASS_DEFAULT} flex-shrink-0" />
						<span>{section.title}</span>
					</a>
				{:else}
					<button
						class="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm whitespace-nowrap transition-colors first:ml-4 last:mr-4 hover:bg-accent {isActive(
							section
						)
							? 'bg-accent text-accent-foreground'
							: 'text-muted-foreground'}"
						{...{ [UI_DATA_ATTRS.ACTIVE]: isActive(section) }}
						onclick={(e: MouseEvent) => {
							onSectionChange?.(section.title);
							carousel.scrollToCenter(e.currentTarget as HTMLElement);
						}}
					>
						<section.icon class="{ICON_CLASS_DEFAULT} flex-shrink-0" />
						<span>{section.title}</span>
					</button>
				{/if}
			{/each}
		</ScrollCarousel>
	</div>
</div>
