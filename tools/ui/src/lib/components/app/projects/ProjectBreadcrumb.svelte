<script lang="ts">
	import { ROUTES, TAB_SURFACE_CLASS, TAB_SURFACE_IDLE_CLASS } from '$lib/constants';
	import { RouterService } from '$lib/services/router.service';
	import type { Project } from '$lib/types';

	interface Props {
		/** The project page is the current page. */
		isCurrent?: boolean;
		project: Project;
		/** Starts from the project list; off where the context makes it obvious. */
		root?: boolean;
		/** Ends with a separator, for items that follow (e.g. tabs). */
		trailingSeparator?: boolean;
	}

	let { isCurrent = false, project, root = true, trailingSeparator = false }: Props = $props();

	let projectHref = $derived(RouterService.project(project.id));
</script>

{#snippet crumb(href: string, label: string, current: boolean)}
	<a
		aria-current={current ? 'page' : undefined}
		class={[
			TAB_SURFACE_CLASS,
			TAB_SURFACE_IDLE_CLASS,
			'min-w-0 px-2',
			current ? 'text-foreground' : 'text-muted-foreground hover:text-foreground'
		]}
		{href}
	>
		<span class="truncate">{label}</span>
	</a>
{/snippet}

<nav aria-label="Breadcrumb" class="flex min-w-0 shrink-0 items-center gap-0.5 text-sm">
	{#if root}
		{@render crumb(ROUTES.PROJECTS, 'Projects', false)}

		<span class="text-muted-foreground">/</span>
	{/if}

	{@render crumb(projectHref, project.name, isCurrent)}

	{#if trailingSeparator}
		<span class="text-muted-foreground">/</span>
	{/if}
</nav>
