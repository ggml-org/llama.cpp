<script lang="ts">
	import SidebarNavigationActions from './SidebarNavigationActions.svelte';
	import { SIDEBAR_FOOTER_ITEMS } from '#lib/constants/index.ts';
	import type { SidebarAction } from '#lib/enums/index.ts';

	interface Props {
		onAction?: (action: SidebarAction) => void;
	}

	let { onAction }: Props = $props();
</script>

<!-- the footer: managing the app around the chats, a group inset in the sidebar, its rows
     rounded concentric with it (see the nested surfaces in app.css) -->
<SidebarNavigationActions
	class="inset-group-surface mx-2 mb-2 p-(--surface-group-padding)"
	items={SIDEBAR_FOOTER_ITEMS}
	{onAction}
/>
