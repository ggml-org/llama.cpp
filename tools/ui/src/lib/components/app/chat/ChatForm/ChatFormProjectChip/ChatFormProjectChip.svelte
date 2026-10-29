<script lang="ts">
	import { X } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app/actions';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { conversationsStore, projectsStore, uiStore } from '$lib/stores';
	import type { Project } from '$lib/types';

	interface Props {
		disabled?: boolean;
		project: Project;
	}

	let { disabled = false, project }: Props = $props();

	function pickProject() {
		uiStore.openProjectPicker({
			currentProjectId: project.id,
			onSelect: (projectId) =>
				void conversationsStore.assignActiveConversationProject(
					projectsStore.get(projectId) ?? null
				)
		});
	}
</script>

<!-- the chat's project, beside its working directory: a project and a folder
     are different things, so each has its own chip. The remove
     button sits beside the chip, not inside it: buttons do not nest -->
<div class="group flex min-w-0 items-center rounded-md px-2 py-1 text-xs backdrop-blur-2xl">
	<Tooltip.Root>
		<Tooltip.Trigger>
			{#snippet child({ props })}
				<button
					{...props}
					class="flex min-w-0 cursor-pointer items-center gap-1 text-foreground"
					{disabled}
					onclick={pickProject}
					type="button"
				>
					<span class="max-w-48 truncate">{project.name}</span>
				</button>
			{/snippet}
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>Change project</p>
		</Tooltip.Content>
	</Tooltip.Root>

	<div
		class="w-0 overflow-hidden opacity-0 transition-[width,opacity] duration-200 ease-out group-focus-within:ml-1 group-focus-within:w-auto group-focus-within:opacity-100 group-hover:ml-1 group-hover:w-auto group-hover:opacity-100"
	>
		<ActionIcon
			ariaLabel="Remove from project"
			class="!h-4 !w-4 shrink-0 text-muted-foreground hover:text-foreground"
			{disabled}
			icon={X}
			iconSize="h-3 w-3"
			onclick={() => void conversationsStore.assignActiveConversationProject(null)}
			tooltip="Remove from project"
		/>
	</div>
</div>
