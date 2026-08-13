<script lang="ts">
	import { SkillCatalog } from '$lib/components/app/skills';
	import { useSkillCatalogRefresh } from '$lib/hooks/use-skill-catalog-refresh.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';

	// The route owns the CWD derivation and the abortable refresh lifecycle.
	// `cwd` reacts to the active conversation's working directory (buffered
	// via pendingCwd before the first chat exists); the refresh controller
	// invalidates the previous route slot on change and aborts on unmount.
	// Frozen agent run snapshots are never altered.
	const refresh = useSkillCatalogRefresh();

	const cwd = $derived(
		conversationsStore.activeConversation?.cwd ?? conversationsStore.pendingCwd ?? undefined
	);

	$effect(() => {
		refresh.onCwdChange(cwd);
	});

	$effect(() => () => refresh.dispose());
</script>

<svelte:head>
	<title>Skills · llama.cpp</title>
</svelte:head>

<SkillCatalog {cwd} onRetry={refresh.retry} />
