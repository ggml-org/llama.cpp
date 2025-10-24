<script lang="ts">
	import { Wrench } from '@lucide/svelte';
	import ChevronsUpDownIcon from '@lucide/svelte/icons/chevrons-up-down';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { buttonVariants } from '$lib/components/ui/button/index.js';
	import { Card } from '$lib/components/ui/card';
	import type { ApiChatCompletionToolCall } from '$lib/types/api';

	interface Props {
		class?: string;
		index: number;
		toolCall: ApiChatCompletionToolCall;
	}

	let { class: className = '', index, toolCall }: Props = $props();

	let isExpanded = $state(false);

	const headerLabel = $derived.by(() => {
		const callNumber = index + 1;
		const functionName = toolCall.function?.name?.trim();

		return functionName ? `Tool call #${callNumber} · ${functionName}` : `Tool call #${callNumber}`;
	});

	const formattedPayload = $derived.by(() => {
		const payload: Record<string, unknown> = {};

		if (toolCall.id) {
			payload.id = toolCall.id;
		}

		if (toolCall.type) {
			payload.type = toolCall.type;
		}

		if (toolCall.function) {
			const fnPayload: Record<string, unknown> = {};
			const { name, arguments: args } = toolCall.function;

			if (name) {
				fnPayload.name = name;
			}

			const trimmedArguments = args?.trim();
			if (trimmedArguments) {
				try {
					fnPayload.arguments = JSON.parse(trimmedArguments);
				} catch {
					fnPayload.arguments = trimmedArguments;
				}
			}

			if (Object.keys(fnPayload).length > 0) {
				payload.function = fnPayload;
			}
		}

		return JSON.stringify(payload, null, 2);
	});
</script>

<Collapsible.Root bind:open={isExpanded} class="mb-3 last:mb-0 {className}">
	<Card class="gap-0 border-muted bg-muted/30 py-0">
		<Collapsible.Trigger class="flex cursor-pointer items-center justify-between p-3">
			<div class="flex items-center gap-2 text-muted-foreground">
				<Wrench class="h-4 w-4" />

				<span class="text-sm font-medium">{headerLabel}</span>
			</div>

			<div
				class={buttonVariants({
					variant: 'ghost',
					size: 'sm',
					class: 'h-6 w-6 p-0 text-muted-foreground hover:text-foreground'
				})}
			>
				<ChevronsUpDownIcon class="h-4 w-4" />

				<span class="sr-only">Toggle tool call payload</span>
			</div>
		</Collapsible.Trigger>

		<Collapsible.Content>
			<div class="border-t border-muted px-3 pb-3">
				<div class="pt-3">
					<pre class="tool-call-content">{formattedPayload}</pre>
				</div>
			</div>
		</Collapsible.Content>
	</Card>
</Collapsible.Root>

<style>
	.tool-call-content {
		font-family: var(--font-mono);
		font-size: 0.75rem;
		line-height: 1.25rem;
		white-space: pre-wrap;
		word-break: break-word;
		margin: 0;
	}
</style>
