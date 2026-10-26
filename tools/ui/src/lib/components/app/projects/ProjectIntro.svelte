<script lang="ts">
	import ProjectActions from './ProjectActions.svelte';
	import { Folder } from '@lucide/svelte';
	import { conversationsStore, projectsStore, toolsStore } from '$lib/stores';
	import type { Project } from '$lib/types';
	import { abbreviateHome, formatRelativeTime } from '$lib/utils';

	interface Props {
		project: Project;
	}

	let { project }: Props = $props();

	let chatCount = $derived(
		conversationsStore.conversations.filter((conversation) => conversation.projectId === project.id)
			.length
	);

	function commitName(event: Event & { currentTarget: HTMLInputElement }) {
		const name = event.currentTarget.value.trim();

		if (name && name !== project.name) {
			void projectsStore.update(project.id, { name });
		} else {
			event.currentTarget.value = project.name;
		}
	}

	function commitDescription(event: Event & { currentTarget: HTMLTextAreaElement }) {
		const description = event.currentTarget.value.trim();

		if (description !== project.description) {
			void projectsStore.update(project.id, { description });
		}
	}
</script>

<div class="flex flex-col gap-2">
	<div class="flex items-center gap-3">
		<input
			aria-label="Project name"
			class="min-w-0 flex-1 truncate border-0 bg-transparent p-0 text-2xl font-semibold tracking-tight outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
			maxlength={200}
			onblur={commitName}
			onkeydown={(event) => {
				if (event.key === 'Enter') event.currentTarget.blur();
			}}
			spellcheck={false}
			value={project.name}
		/>

		<ProjectActions {project} />
	</div>

	<!-- the goal is edited in place, like the name; Enter commits, Shift+Enter breaks a line -->
	<textarea
		aria-label="Project goal"
		class="field-sizing-content min-h-0 w-full max-w-2xl resize-none border-0 bg-transparent p-0 text-sm text-muted-foreground outline-none placeholder:text-muted-foreground/60 focus:text-foreground"
		maxlength={500}
		onblur={commitDescription}
		onkeydown={(event) => {
			if (event.key === 'Enter' && !event.shiftKey) {
				event.preventDefault();
				event.currentTarget.blur();
			}
		}}
		placeholder="What are you trying to achieve?"
		rows={1}
		value={project.description}
	></textarea>

	<p class="flex min-w-0 flex-wrap items-center gap-x-1.5 text-xs text-muted-foreground">
		<span>{chatCount} {chatCount === 1 ? 'chat' : 'chats'}</span>

		<span aria-hidden="true">·</span>

		<span>updated {formatRelativeTime(project.updatedAt)}</span>

		{#if project.cwd}
			<span aria-hidden="true">·</span>

			<span class="inline-flex min-w-0 items-center gap-1" title={project.cwd}>
				<Folder class="h-3 w-3 shrink-0" />

				<span class="truncate">{abbreviateHome(project.cwd, toolsStore.serverHome)}</span>
			</span>
		{/if}
	</p>
</div>
