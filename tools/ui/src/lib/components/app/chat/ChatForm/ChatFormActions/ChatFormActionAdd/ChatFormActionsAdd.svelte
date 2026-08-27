<script lang="ts">
	import ChatFormActionAddButton from './ChatFormActionAddButton.svelte';
	import ChatFormActionAddDropdown from './ChatFormActionAddDropdown.svelte';
	import ChatFormActionAddSheet from './ChatFormActionAddSheet.svelte';
	import { deviceStore } from '$lib/stores';
</script>

{#if deviceStore.isMobile}
	<ChatFormActionAddSheet>
		{#snippet trigger({ disabled, onclick })}
			<ChatFormActionAddButton {disabled} {onclick} />
		{/snippet}
	</ChatFormActionAddSheet>
{:else}
	<ChatFormActionAddDropdown />
{/if}
