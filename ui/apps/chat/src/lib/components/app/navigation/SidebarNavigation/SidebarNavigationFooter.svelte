<script lang="ts">
	import SidebarNavigationActions from './SidebarNavigationActions.svelte';
	import { SIDEBAR_FOOTER_ITEMS } from '#lib/constants/index.ts';
	import type { SidebarAction } from '#lib/enums/index.ts';
	import { uiStore } from '#lib/stores/index.ts';

	interface Props {
		onAction?: (action: SidebarAction) => void;
	}

	let { onAction }: Props = $props();

	// a lone row needs no group around it
	let isGrouped = $derived(
		SIDEBAR_FOOTER_ITEMS.filter((item) => uiStore.isSidebarActionShown(item.action)).length > 1
	);
</script>

<!-- the footer: managing the app around the chats. Two or more rows are a group inset in the
     sidebar, rounded concentric with it (see the nested surfaces in app.css); a single row sits
     on the sidebar as the top actions do -->
<SidebarNavigationActions
	class={isGrouped ? 'inset-group-surface mx-2 mb-2 p-(--surface-group-padding)' : 'mb-2 px-2'}
	items={SIDEBAR_FOOTER_ITEMS}
	{onAction}
/>
