<script lang="ts">
	import MenuActionsList from './MenuActionsList.svelte';
	import * as ContextMenu from '$lib/components/ui/context-menu';
	import { MenuKind } from '$lib/enums';
	import { deviceStore } from '$lib/stores';
	import type { MenuAction } from '$lib/types';
	import type { Snippet } from 'svelte';

	interface Props {
		actions: MenuAction[];
		children: Snippet;
		/** Leave right-click to the browser, e.g. while the item is being selected. */
		disabled?: boolean;
		open?: boolean;
	}

	let { actions, children, disabled = false, open = $bindable(false) }: Props = $props();
</script>

<!-- the trigger takes no box of its own, so the item lays out as if unwrapped; a
     phone keeps the browser's long-press and reaches the actions from the item's
     "more actions" drawer instead -->
<ContextMenu.Root bind:open>
	<ContextMenu.Trigger class="contents" disabled={disabled || deviceStore.isMobile}>
		{@render children()}
	</ContextMenu.Trigger>

	<ContextMenu.Content class="z-[999999] w-48">
		<MenuActionsList {actions} kind={MenuKind.CONTEXT} onClose={() => (open = false)} />
	</ContextMenu.Content>
</ContextMenu.Root>
