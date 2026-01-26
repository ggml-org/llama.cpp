<script lang="ts">
	import { Wrench, Database, MessageSquare, FileText, Sparkles, ListChecks } from '@lucide/svelte';
	import { cn } from '$lib/components/ui/utils';
	import type { MCPCapabilitiesInfo } from '$lib/types';
	import { Badge } from '$lib/components/ui/badge';

	interface Props {
		capabilities?: MCPCapabilitiesInfo;
		class?: string;
	}

	let { capabilities, class: className }: Props = $props();
</script>

{#if capabilities}
	<div class={cn('space-y-2 text-xs', className)}>
		<div>
			<div class="flex flex-wrap gap-2">
				{#if capabilities.server.tools}
					<Badge
						variant="outline"
						class="h-5 gap-1 bg-green-50 px-1.5 text-[10px] dark:bg-green-950"
					>
						<Wrench class="h-3 w-3 text-green-600 dark:text-green-400" />

						Tools
					</Badge>
				{/if}

				{#if capabilities.server.resources}
					<Badge variant="outline" class="h-5 gap-1 bg-blue-50 px-1.5 text-[10px] dark:bg-blue-950">
						<Database class="h-3 w-3 text-blue-600 dark:text-blue-400" />

						Resources
					</Badge>
				{/if}

				{#if capabilities.server.prompts}
					<Badge
						variant="outline"
						class="h-5 gap-1 bg-purple-50 px-1.5 text-[10px] dark:bg-purple-950"
					>
						<MessageSquare class="h-3 w-3 text-purple-600 dark:text-purple-400" />

						Prompts
					</Badge>
				{/if}

				{#if capabilities.server.logging}
					<Badge
						variant="outline"
						class="h-5 gap-1 bg-orange-50 px-1.5 text-[10px] dark:bg-orange-950"
					>
						<FileText class="h-3 w-3 text-orange-600 dark:text-orange-400" />

						Logging
					</Badge>
				{/if}

				{#if capabilities.server.completions}
					<Badge variant="outline" class="h-5 gap-1 bg-cyan-50 px-1.5 text-[10px] dark:bg-cyan-950">
						<Sparkles class="h-3 w-3 text-cyan-600 dark:text-cyan-400" />

						Completions
					</Badge>
				{/if}

				{#if capabilities.server.tasks}
					<Badge variant="outline" class="h-5 gap-1 bg-pink-50 px-1.5 text-[10px] dark:bg-pink-950">
						<ListChecks class="h-3 w-3 text-pink-600 dark:text-pink-400" />

						Tasks
					</Badge>
				{/if}
			</div>
		</div>
	</div>
{/if}
