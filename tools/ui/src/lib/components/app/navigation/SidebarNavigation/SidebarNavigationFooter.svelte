<script lang="ts">
	import SidebarNavigationActions from './SidebarNavigationActions.svelte';
	import { SIDEBAR_FOOTER_ITEMS } from '$lib/constants';
	import type { SidebarAction } from '$lib/enums';

	interface Props {
		onAction?: (action: SidebarAction) => void;
	}

	let { onAction }: Props = $props();
</script>

<!-- the footer: managing the app around the chats, a group inset in the sidebar, its rows
     rounded concentric with it (see the nested surfaces in app.css) -->
<SidebarNavigationActions
	class="mx-2 mb-2 rounded-(--surface-group-radius) border border-border/20 bg-background/55 p-(--surface-group-padding) shadow-sm dark:bg-foreground/5"
	items={SIDEBAR_FOOTER_ITEMS}
	{onAction}
/>
