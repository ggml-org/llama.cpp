<script lang="ts">
	import ProjectContext from './ProjectContext.svelte';
	import ProjectInstructions from './ProjectInstructions.svelte';
	import * as Card from '$lib/components/ui/card';
	import { FileMentionEntryType, ProjectContextKind } from '$lib/enums';
	import { useChatScreenActiveModel } from '$lib/hooks/use-chat-screen-active-model.svelte';
	import { projectsStore } from '$lib/stores';
	import type { FileMentionEntry, Project, ProjectContextItem } from '$lib/types';
	import { uuid } from '$lib/utils';
	import { parseFilesToMessageExtras, processFilesToChatUploaded } from '$lib/utils/browser-only';

	interface Props {
		project: Project;
	}

	let { project }: Props = $props();

	const activeModel = useChatScreenActiveModel();

	async function addContextItems(items: Omit<ProjectContextItem, 'id'>[]) {
		if (items.length === 0) return;

		await projectsStore.update(project.id, {
			contextItems: [...project.contextItems, ...items.map((item) => ({ ...item, id: uuid() }))]
		});
	}

	/**
	 * Uploaded project files are parsed the same way chat attachments are, so a
	 * project chat can put their content in front of the model.
	 */
	async function handleAddAttachments(files: File[]) {
		const modelId = activeModel.activeModelId ?? undefined;
		const uploaded = await processFilesToChatUploaded(files, modelId);
		const { extras } = await parseFilesToMessageExtras($state.snapshot(uploaded), modelId);

		await addContextItems(
			extras.map((extra) => ({
				extra,
				kind: ProjectContextKind.ATTACHMENT,
				name: 'name' in extra ? String(extra.name) : 'attachment',
				size: 'size' in extra ? extra.size : undefined
			}))
		);
	}

	/** A file or MCP resource picked with `@` is kept as a reference. */
	async function handleAddReference(entry: FileMentionEntry) {
		await addContextItems([
			{
				kind:
					entry.type === FileMentionEntryType.MCP_RESOURCE
						? ProjectContextKind.MCP_RESOURCE
						: ProjectContextKind.FILE,
				name: entry.name,
				path: entry.path,
				serverName: entry.serverName
			}
		]);
	}
</script>

<!-- instructions and context side by side, each in a card of its own -->
<div class="grid gap-4 md:grid-cols-2">
	<Card.Root class="gap-0 bg-muted/30 p-0">
		<ProjectInstructions
			instructions={project.instructions}
			onSave={(instructions) => void projectsStore.update(project.id, { instructions })}
		/>
	</Card.Root>

	<Card.Root class="gap-0 bg-muted/30 p-0">
		<ProjectContext
			items={project.contextItems}
			onAddAttachments={(files) => void handleAddAttachments(files)}
			onAddReference={(entry) => void handleAddReference(entry)}
			onRemove={(itemId) =>
				void projectsStore.update(project.id, {
					contextItems: project.contextItems.filter((item) => item.id !== itemId)
				})}
		/>
	</Card.Root>
</div>
