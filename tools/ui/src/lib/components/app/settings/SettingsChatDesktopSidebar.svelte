<script lang="ts">
	import { Settings } from '@lucide/svelte';
	import type { SettingsSection, SettingsSectionTitle } from '$lib/constants';

	interface Props {
		sections: SettingsSection[];
		isActive: (section: SettingsSection) => boolean;
		getHref?: (section: SettingsSection) => string;
		onSectionChange?: (section: SettingsSectionTitle) => void;
	}

	let { sections, isActive, getHref, onSectionChange }: Props = $props();
</script>

<div class="sticky top-2 hidden w-64 flex-col self-start bg-background py-4 md:flex gap-6">
	<div class="flex items-center gap-2 py-2">
		<Settings class="h-5 w-5 md:h-6 md:w-6" />

		<h1 class="text-xl font-semibold md:text-2xl">Settings</h1>
	</div>

	<nav class="space-y-1">
		{#each sections as section (section.title)}
			{#if getHref}
				<a
					class="flex w-full cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-left text-sm no-underline transition-colors hover:bg-accent {isActive(
						section
					)
						? 'bg-accent text-accent-foreground'
						: 'text-muted-foreground'}"
					href={getHref(section)}
				>
					<section.icon class="h-4 w-4" />
					<span class="ml-2">{section.title}</span>
				</a>
			{:else}
				<button
					class="flex w-full cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors hover:bg-accent {isActive(
						section
					)
						? 'bg-accent text-accent-foreground'
						: 'text-muted-foreground'}"
					onclick={() => onSectionChange?.(section.title)}
				>
					<section.icon class="h-4 w-4" />
					<span class="ml-2">{section.title}</span>
				</button>
			{/if}
		{/each}
	</nav>
</div>
