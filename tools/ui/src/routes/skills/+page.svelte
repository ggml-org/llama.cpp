<script lang="ts">
	import { SkillCatalog } from '$lib/components/app/skills';
	import { useSkillCatalogRefresh } from '$lib/hooks/use-skill-catalog-refresh.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';

	// Route-owned CWD refresh; stale responses are invalidated and snapshots stay immutable.
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
