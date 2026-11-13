<script lang="ts">
	import SidebarNavigationConversationItem from '$lib/components/app/navigation/SidebarNavigation/SidebarNavigationConversationItem.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import type { ComponentProps } from 'svelte';

	let props: ComponentProps<typeof SidebarNavigationConversationItem> = $props();
</script>

<!-- the row's menu sits in a tooltip, whose provider the app layout mounts -->
<Tooltip.Provider>
	<SidebarNavigationConversationItem {...props} />
</Tooltip.Provider>
