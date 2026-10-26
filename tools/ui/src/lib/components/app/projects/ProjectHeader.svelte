<script lang="ts">
	import ProjectBreadcrumb from './ProjectBreadcrumb.svelte';
	import type { Project } from '$lib/types';

	interface Props {
		project: Project;
	}

	let { project }: Props = $props();
</script>

<!-- the way back to the projects list; the project's own actions sit by its title -->
<header class="flex items-center">
	<div class="-ml-2 flex min-w-0 flex-1">
		<ProjectBreadcrumb isCurrent {project} />
	</div>
</header>
