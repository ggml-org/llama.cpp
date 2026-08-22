<script lang="ts">
	import { ScrollCarousel } from '$lib/components/app';
	import { ICON_CLASS_DEFAULT, UI_DATA_ATTRS } from '$lib/constants';
	import { BooleanString } from '$lib/enums';
	import { useScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';
	import type { SettingsSection, SettingsSectionTitle } from '$lib/types';
	import { onMount, tick } from 'svelte';

	interface Props {
		sections: SettingsSection[];
		isActive: (section: SettingsSection) => boolean;
		onSectionChange?: (section: SettingsSectionTitle) => void;
	}

	let { isActive, onSectionChange, sections }: Props = $props();

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

<div class="flex flex-col bg-background md:hidden sticky top-13 z-50">
	<div class="border-b border-border/30">
		<ScrollCarousel {carousel} alwaysShowArrows containerClass="py-2" innerClass="gap-2">
			{#each sections as section (section.title)}
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
			{/each}
		</ScrollCarousel>
	</div>
</div>
