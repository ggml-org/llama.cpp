<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import type { SettingsSection, SettingsSectionTitle } from '$lib/types';

	interface Props {
		sections: SettingsSection[];
		isActive: (section: SettingsSection) => boolean;
		onSectionChange?: (section: SettingsSectionTitle) => void;
	}

	let { isActive, onSectionChange, sections }: Props = $props();
</script>

<div class="sticky top-12 hidden w-64 flex-col self-start bg-background md:flex gap-6">
	<nav class="space-y-1">
		{#each sections as section (section.title)}
			<button
				class="flex w-full cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors hover:bg-accent {isActive(
					section
				)
					? 'bg-accent text-accent-foreground'
					: 'text-muted-foreground'}"
				onclick={() => onSectionChange?.(section.title)}
			>
				<section.icon class={ICON_CLASS_DEFAULT} />
				<span class="ml-2">{section.title}</span>
			</button>
		{/each}
	</nav>
</div>
