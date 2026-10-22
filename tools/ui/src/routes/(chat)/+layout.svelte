<script lang="ts">
	import { page } from '$app/state';
	import {
		ChatScreen,
		ChatTabs,
		ProjectHeader,
		ProjectIntro,
		ProjectOverview,
		ProjectPanels
	} from '$lib/components/app';
	import { NEW_CHAT_TAB_ID } from '$lib/constants';
	import { projectsStore, settingsStore, tabsStore } from '$lib/stores';

	let { children } = $props();

	// the project home shows the project parts, and no tab bar
	let project = $derived(
		page.params.projectId ? projectsStore.get(page.params.projectId) : undefined
	);

	// the new-chat screen is the bare `#/` route (no conversation id)
	let showCenteredEmpty = $derived(!page.params.id && !page.params.projectId);

	let showTabs = $derived(
		Boolean(settingsStore.config.conversationTabs) &&
			!page.params.projectId &&
			(page.params.id || tabsStore.openTabs.some((id) => id !== NEW_CHAT_TAB_ID))
	);

	$effect(() => {
		const id = page.params.id ?? (page.route.id === '/(chat)' ? NEW_CHAT_TAB_ID : undefined);

		if (id && settingsStore.config.conversationTabs) {
			tabsStore.syncWithRoute(id);
		}
	});
</script>

{#snippet projectHeader()}
	{#if project}
		<ProjectHeader {project} />
	{/if}
{/snippet}

{#snippet projectIntro()}
	{#if project}
		<ProjectIntro {project} />
	{/if}
{/snippet}

{#snippet projectPanels()}
	{#if project}
		<ProjectPanels {project} />
	{/if}
{/snippet}

{#snippet projectOverview()}
	{#if project}
		<ProjectOverview {project} />
	{/if}
{/snippet}

<!-- the offset is the tab bar's full height (pt-3.25 + h-10), so the chat
     column still fits the viewport and only scrolls when the messages overflow -->
<div class={showTabs ? 'md:[--chat-tabs-height:3.3125rem] md:[--chat-tabs-offset:3.3125rem]' : ''}>
	{#if showTabs}
		<ChatTabs />
	{/if}

	<ChatScreen
		header={project ? projectHeader : undefined}
		intro={project ? projectIntro : undefined}
		overview={project ? projectOverview : undefined}
		panels={project ? projectPanels : undefined}
		{showCenteredEmpty}
	/>
</div>

{@render children?.()}
