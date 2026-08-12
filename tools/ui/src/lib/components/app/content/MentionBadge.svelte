<script lang="ts">
	import { SETTINGS_KEYS } from '$lib/constants';
	import { settingsStore } from '$lib/stores/settings.svelte';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import {
		getMentionBadgeIconPaths,
		getMentionBadgeLabel,
		MENTION_BADGE_CLASSNAME,
		MENTION_BADGE_ICON_CLASSNAME,
		MENTION_BADGE_SVG_ATTRIBUTES
	} from '$lib/utils';

	interface Props {
		name: string;
		path: string;
	}

	let { name, path }: Props = $props();

	let showFullPath = $derived(
		settingsStore.getConfig(SETTINGS_KEYS.SHOW_FULL_PATH_IN_MENTIONS) as boolean
	);
	let label = $derived(getMentionBadgeLabel(name, path, showFullPath, toolsStore.serverHome));
</script>

<!-- The chip is a flex container, so template whitespace between its
     children collapses away and the icon keeps its `gap-1` spacing. -->
<span class={MENTION_BADGE_CLASSNAME} title={path}>
	<svg {...MENTION_BADGE_SVG_ATTRIBUTES} class={MENTION_BADGE_ICON_CLASSNAME}>
		{#each getMentionBadgeIconPaths(path) as d (d)}
			<path {d} />
		{/each}
	</svg>

	<span class="shrink-0 truncate">{label}</span>
</span>
