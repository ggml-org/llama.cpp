<script lang="ts">
	import { projectsStore } from '$lib/stores';
	import type { Project } from '$lib/types';

	interface Props {
		project: Project;
	}

	let { project }: Props = $props();

	function commitName(event: Event & { currentTarget: HTMLInputElement }) {
		const name = event.currentTarget.value.trim();

		if (name && name !== project.name) {
			void projectsStore.update(project.id, { name });
		} else {
			event.currentTarget.value = project.name;
		}
	}
</script>

<div class="flex flex-col gap-3">
	<input
		aria-label="Project name"
		class="min-w-0 truncate border-0 bg-transparent p-0 text-2xl font-semibold tracking-tight outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
		maxlength={200}
		onblur={commitName}
		onkeydown={(event) => {
			if (event.key === 'Enter') event.currentTarget.blur();
		}}
		spellcheck={false}
		value={project.name}
	/>

	{#if project.description}
		<p class="max-w-2xl text-sm text-balance text-muted-foreground">{project.description}</p>
	{/if}
</div>
