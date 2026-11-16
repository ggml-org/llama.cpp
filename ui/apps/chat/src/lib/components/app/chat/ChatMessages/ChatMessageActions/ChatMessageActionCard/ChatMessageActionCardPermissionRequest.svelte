<script lang="ts">
	import { ChevronDown, ShieldQuestion } from '@lucide/svelte';
	import { ChatMessageActionCard } from '#lib/components/app/index.ts';
	import { Button, buttonVariants } from '#lib/components/ui/button/index.ts';
	import * as ButtonGroup from '#lib/components/ui/button-group/index.ts';
	import * as DropdownMenu from '#lib/components/ui/dropdown-menu/index.ts';
	import { cn } from '#lib/components/ui/utils.ts';
	import { TOOL_SERVER_LABELS } from '#lib/constants/index.ts';
	import { ToolPermissionDecision, ToolSource } from '#lib/enums/index.ts';
	import { toolsStore } from '#lib/stores/index.ts';

	interface Props {
		toolName: string;
		serverLabel: string;
		onDecision: (decision: ToolPermissionDecision) => void;
	}

	let { onDecision, serverLabel, toolName }: Props = $props();
</script>

<ChatMessageActionCard icon={ShieldQuestion}>
	{#snippet message()}
		Allow use of <span class="font-semibold">{toolName}</span>{#if serverLabel}
			&nbsp;from <span class="font-semibold">{serverLabel}</span>{/if}?
	{/snippet}

	{#snippet actions()}
		<DropdownMenu.Root>
			<ButtonGroup.Root class="overflow-hidden rounded-md shadow-sm">
				<Button
					class="!rounded-r-none !shadow-none"
					onclick={() => onDecision(ToolPermissionDecision.ONCE)}
					size="sm"
					variant="secondary"
				>
					Allow once
				</Button>

				<ButtonGroup.Separator />

				<DropdownMenu.Trigger
					aria-label="More allow options"
					class={cn(
						buttonVariants({ size: 'sm', variant: 'secondary' }),
						'inline-flex cursor-pointer items-center !rounded-l-none !shadow-none !px-2'
					)}
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
						source === ToolSource.SERVER
							? TOOL_SERVER_LABELS[ToolSource.SERVER]
							: source === ToolSource.CUSTOM
								? TOOL_SERVER_LABELS[ToolSource.CUSTOM]
								: 'MCP Tools'}
					<DropdownMenu.Item onclick={() => onDecision(ToolPermissionDecision.ALWAYS_SERVER)}>
						Approve all tools from {providerName}
					</DropdownMenu.Item>
				{/if}
			</DropdownMenu.Content>
		</DropdownMenu.Root>

		<Button onclick={() => onDecision(ToolPermissionDecision.DENY)} size="sm" variant="destructive">
			Deny
		</Button>
	{/snippet}
</ChatMessageActionCard>
