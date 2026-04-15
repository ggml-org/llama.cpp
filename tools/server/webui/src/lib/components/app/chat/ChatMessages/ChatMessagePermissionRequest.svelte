<script lang="ts">
	import { ChevronDown, ShieldQuestion } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as ButtonGroup from '$lib/components/ui/button-group';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { ToolSource, ToolPermissionDecision, ToolServerLabel } from '$lib/enums';
	import { toolsStore } from '$lib/stores/tools.svelte';

	interface Props {
		toolName: string;
		serverLabel: string;
		onDecision: (decision: ToolPermissionDecision) => void;
	}

	let { toolName, serverLabel, onDecision }: Props = $props();
</script>

<div class="permission-request my-2 rounded-lg border border-border bg-card p-3">
	<div class="mb-3 flex items-center gap-2 text-sm">
		<ShieldQuestion class="h-4 w-4 shrink-0 text-muted-foreground" />
		<span>
			Allow use of
			<span class="font-semibold">{toolName}</span>
			{#if serverLabel}
				from <span class="font-semibold">{serverLabel}</span>
			{/if}
			?
		</span>
	</div>
	<div class="flex flex-wrap items-center gap-2">
		<DropdownMenu.Root>
			<ButtonGroup.Root
				class="overflow-hidden rounded-md bg-foreground text-white shadow-sm dark:bg-secondary dark:text-foreground"
			>
				<Button
					class="rounded-none! shadow-none!"
					size="sm"
					onclick={() => onDecision(ToolPermissionDecision.ONCE)}
				>
					Allow once
				</Button>

				<ButtonGroup.Separator />

				<DropdownMenu.Trigger>
					<Button size="sm" class="rounded-none! !ps-2 shadow-none!">
						<ChevronDown class="h-3.5 w-3.5" />
					</Button>
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
							? ToolServerLabel.BUILTIN
							: source === ToolSource.CUSTOM
								? ToolServerLabel.CUSTOM
								: 'MCP Tools'}
					<DropdownMenu.Item onclick={() => onDecision(ToolPermissionDecision.ALWAYS_SERVER)}>
						Approve all tools from {providerName}
					</DropdownMenu.Item>
				{/if}
			</DropdownMenu.Content>
		</DropdownMenu.Root>

		<Button
			variant="destructive"
			size="sm"
			class="text-destructive hover:text-destructive"
			onclick={() => onDecision(ToolPermissionDecision.DENY)}
		>
			Deny
		</Button>
	</div>
</div>
