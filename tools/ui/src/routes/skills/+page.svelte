<script lang="ts">
	import { SkillCatalog } from '$lib/components/app/skills';
	import { conversationsStore } from '$lib/stores/conversations/index.svelte';
	import { skillsStore } from '$lib/stores/skills.svelte';

	// Route-owned CWD refresh; stale responses are invalidated and snapshots stay immutable.
	const cwd = $derived(
		conversationsStore.activeConversation?.cwd ??
			conversationsStore.preferences.pendingCwd ??
			undefined
	);

	$effect(() => {
		skillsStore.onRouteCwdChange(cwd);
	});

	$effect(() => () => skillsStore.disposeRouteCatalog());
</script>

<svelte:head>
	<title>Skills · llama.cpp</title>
</svelte:head>

<SkillCatalog {cwd} onRetry={() => skillsStore.retryRouteCatalog()} />
