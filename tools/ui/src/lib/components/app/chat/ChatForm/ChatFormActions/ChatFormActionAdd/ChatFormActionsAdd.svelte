<script lang="ts">
	import ChatFormActionAddButton from './ChatFormActionAddButton.svelte';
	import ChatFormActionAddDrawer from './ChatFormActionAddDrawer.svelte';
	import ChatFormActionAddDropdown from './ChatFormActionAddDropdown.svelte';
	import { deviceStore } from '$lib/stores';
</script>

{#if deviceStore.isMobile}
	<ChatFormActionAddDrawer>
		{#snippet trigger({ disabled, onclick })}
			<ChatFormActionAddButton {disabled} {onclick} />
		{/snippet}
	</ChatFormActionAddDrawer>
{:else}
	<ChatFormActionAddDropdown />
{/if}
