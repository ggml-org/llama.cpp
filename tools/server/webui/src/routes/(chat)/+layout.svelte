<script lang="ts">
	import { page } from '$app/state';
	import { ChatScreen } from '$lib/components/app';

	let { children } = $props();

	let showCenteredEmpty = $derived(!page.params.id);
</script>

<ChatScreen {showCenteredEmpty} />

{@render children?.()}
