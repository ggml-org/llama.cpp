<script lang="ts">
	import { ChevronDown, ShieldQuestion } from '@lucide/svelte';
	import { ChatMessageActionCard } from '$lib/components/app';
	import { Button, buttonVariants } from '$lib/components/ui/button';
	import * as ButtonGroup from '$lib/components/ui/button-group';
	import { cn } from '$lib/components/ui/utils';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { ToolSource, ToolPermissionDecision } from '$lib/enums';
	import { TOOL_SERVER_LABELS } from '$lib/constants';
	import { toolsStore } from '$lib/stores/tools.svelte';

	interface Props {
		toolName: string;
		serverLabel: string;
		onDecision: (decision: ToolPermissionDecision) => void;
	}

	let { toolName, serverLabel, onDecision }: Props = $props();
</script>

<ChatMessageActionCard icon={ShieldQuestion}>
	{#snippet message()}
		Allow use of <span class="font-semibold">{toolName}</span>{#if serverLabel}
			from <span class="font-semibold">{serverLabel}</span>{/if}?
	{/snippet}

	{#snippet actions()}
		<DropdownMenu.Root>
			<ButtonGroup.Root class="overflow-hidden rounded-md shadow-sm">
				<Button
					variant="secondary"
					size="sm"
					class="!rounded-r-none !shadow-none"
					onclick={() => onDecision(ToolPermissionDecision.ONCE)}
				>
					Allow once
				</Button>

				<ButtonGroup.Separator />

				<DropdownMenu.Trigger
					class={cn(
						buttonVariants({ variant: 'secondary', size: 'sm' }),
						'inline-flex cursor-pointer items-center !rounded-l-none !shadow-none !px-2'
					)}
					aria-label="More allow options"
				>
					<ChevronDown class="h-3.5 w-3.5" />
				</DropdownMenu.Trigger>
			</ButtonGroup.Root>

			<DropdownMenu.Content align="start" class="min-w-[8rem]">
				<DropdownMenu.Item onclick={() => onDecision(ToolPermissionDecision.ALWAYS)}>
					Always allow <pre>{toolName}</pre>
					tool
				</DropdownMenu.Item>
				{#if serverLabel}
					<DropdownMenu.Item onclick={() => onDecision(ToolPermissionDecision.ALWAYS_SERVER)}>
						Always allow all tools from {serverLabel}
					</DropdownMenu.Item>
				{:else}
					{@const source = toolsStore.getToolSource(toolName)}
					{@const providerName =
						source === ToolSource.BUILTIN
							? TOOL_SERVER_LABELS[ToolSource.BUILTIN]
							: source === ToolSource.CUSTOM
								? TOOL_SERVER_LABELS[ToolSource.CUSTOM]
								: 'MCP Tools'}
					<DropdownMenu.Item onclick={() => onDecision(ToolPermissionDecision.ALWAYS_SERVER)}>
						Approve all tools from {providerName}
					</DropdownMenu.Item>
				{/if}
			</DropdownMenu.Content>
		</DropdownMenu.Root>

		<Button variant="destructive" size="sm" onclick={() => onDecision(ToolPermissionDecision.DENY)}>
			Deny
		</Button>
	{/snippet}
</ChatMessageActionCard>
