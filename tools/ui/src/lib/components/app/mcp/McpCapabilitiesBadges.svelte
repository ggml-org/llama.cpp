<script lang="ts">
	import {
		Database,
		ExternalLink,
		FileText,
		ListChecks,
		MessageSquare,
		Sparkles,
		Wrench
	} from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import type { MCPCapabilitiesInfo } from '$lib/types';

	interface Props {
		capabilities?: MCPCapabilitiesInfo;
		onBrowseResources?: () => void;
	}

	let { capabilities, onBrowseResources }: Props = $props();
</script>

{#if capabilities}
	{#if capabilities.server.tools}
		<Badge class="h-5 gap-1 bg-green-50 px-1.5 text-[10px] dark:bg-green-950" variant="outline">
			<Wrench class="h-3 w-3 text-green-600 dark:text-green-400" />

			Tools
		</Badge>
	{/if}

	{#if capabilities.server.resources}
		<Badge
			class="h-5 cursor-pointer gap-1 bg-blue-50 px-1.5 text-[10px] transition-colors hover:bg-blue-100 dark:bg-blue-950 dark:hover:bg-blue-900"
			onclick={onBrowseResources}
			onkeydown={(e) => {
				if (e.key === 'Enter' || e.key === ' ') {
					e.preventDefault();
					onBrowseResources?.();
				}
			}}
			role="button"
			tabindex={0}
			variant="outline"
		>
			<Database class="h-3 w-3 text-blue-600 dark:text-blue-400" />

			Resources

			<ExternalLink class="h-3 w-3 text-blue-600 dark:text-blue-400" />
		</Badge>
	{/if}

	{#if capabilities.server.prompts}
		<Badge class="h-5 gap-1 bg-purple-50 px-1.5 text-[10px] dark:bg-purple-950" variant="outline">
			<MessageSquare class="h-3 w-3 text-purple-600 dark:text-purple-400" />

			Prompts
		</Badge>
	{/if}

	{#if capabilities.server.logging}
		<Badge class="h-5 gap-1 bg-orange-50 px-1.5 text-[10px] dark:bg-orange-950" variant="outline">
			<FileText class="h-3 w-3 text-orange-600 dark:text-orange-400" />

			Logging
		</Badge>
	{/if}

	{#if capabilities.server.completions}
		<Badge class="h-5 gap-1 bg-cyan-50 px-1.5 text-[10px] dark:bg-cyan-950" variant="outline">
			<Sparkles class="h-3 w-3 text-cyan-600 dark:text-cyan-400" />

			Completions
		</Badge>
	{/if}

	{#if capabilities.server.tasks}
		<Badge class="h-5 gap-1 bg-pink-50 px-1.5 text-[10px] dark:bg-pink-950" variant="outline">
			<ListChecks class="h-3 w-3 text-pink-600 dark:text-pink-400" />

			Tasks
		</Badge>
	{/if}
{/if}
