<script lang="ts">
	import SidebarNavigationActions from './SidebarNavigationActions.svelte';
	import { sidebarFooterItems } from '#lib/constants/index.ts';
	import type { SidebarAction } from '#lib/enums/index.ts';
	import { agenticMode } from '#lib/stores/agentic-mode.svelte.ts';

	interface Props {
		onAction?: (action: SidebarAction) => void;
	}

	let { onAction }: Props = $props();
</script>

<!-- the footer: managing the app around the chats, a group inset in the sidebar, its rows
     rounded concentric with it (see the nested surfaces in app.css) -->
<SidebarNavigationActions
	class="inset-group-surface mx-2 mb-2 p-(--surface-group-padding)"
	items={sidebarFooterItems(agenticMode.enabled)}
	{onAction}
/>
