<script lang="ts">
	import { page } from '$app/state';
	import { ChatScreen } from '$lib/components/app';
	import { encryptionStore } from '$lib/stores/encryption.svelte';

	let { children } = $props();

	let showCenteredEmpty = $derived(!page.params.id);
</script>

<!-- remount on lock to drop the unsent draft held in component state -->
{#key encryptionStore.needsUnlock}
	<ChatScreen {showCenteredEmpty} />
{/key}

{@render children?.()}
