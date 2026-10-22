<script lang="ts">
	import { page } from '$app/state';
	import {
		ChatScreen,
		ChatTabs,
		ProjectBreadcrumb,
		ProjectHeader,
		ProjectIntro,
		ProjectOverview,
		ProjectPanels
	} from '$lib/components/app';
	import { NEW_CHAT_TAB_ID, PROJECT_HOME_TAB_ID } from '$lib/constants';
	import { projectsStore, settingsStore, tabsStore } from '$lib/stores';
	import { resolveTabTarget } from '$lib/utils';

	let { children } = $props();

	let tabTarget = $derived(resolveTabTarget(page));

	let routeProject = $derived(
		page.params.projectId ? projectsStore.get(page.params.projectId) : undefined
	);

	// the project home shows the project parts, and no tab bar
	let project = $derived(tabTarget === PROJECT_HOME_TAB_ID ? routeProject : undefined);

	// chats under `#/projects/<id>/chat/` lead their tab bar with a way back to the project
	let viewProject = $derived(tabTarget === PROJECT_HOME_TAB_ID ? undefined : routeProject);

	// new-chat screens (`#/` or a project's) have no conversation id
	let showCenteredEmpty = $derived(!page.params.id && tabTarget !== PROJECT_HOME_TAB_ID);

	let tabsEnabled = $derived(Boolean(settingsStore.config.conversationTabs));

	let showTabs = $derived(
		tabsEnabled &&
			tabTarget !== null &&
			!project &&
			(!!page.params.id || !!viewProject || tabsStore.openTabs.some((id) => id !== NEW_CHAT_TAB_ID))
	);

	$effect(() => {
		if (tabTarget && tabsEnabled) {
			tabsStore.syncWithRoute(tabTarget);
		}
	});
</script>

{#snippet viewProjectBreadcrumb()}
	{#if viewProject}
		<ProjectBreadcrumb project={viewProject} trailingSeparator />
	{/if}
{/snippet}

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
		<ChatTabs leading={viewProject ? viewProjectBreadcrumb : undefined} />
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
