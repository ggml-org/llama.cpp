<script lang="ts">
	import { CheckCircle2, Circle, ListChecks, Loader2, XCircle } from '@lucide/svelte';
	import type { Component } from 'svelte';
	import type { AgenticTodoItem, AgenticTodoStatus } from '$lib/types/agentic';

	interface Props {
		todos: AgenticTodoItem[];
	}

	let { todos }: Props = $props();

	const completedCount = $derived(todos.filter((todo) => todo.status === 'completed').length);

	function statusIcon(status: AgenticTodoStatus): Component {
		switch (status) {
			case 'completed':
				return CheckCircle2;
			case 'in_progress':
				return Loader2;
			case 'cancelled':
				return XCircle;
			default:
				return Circle;
		}
	}

	function statusClass(status: AgenticTodoStatus): string {
		switch (status) {
			case 'completed':
				return 'text-green-600 dark:text-green-400';
			case 'in_progress':
				return 'text-blue-600 dark:text-blue-400';
			case 'cancelled':
				return 'text-destructive';
			default:
				return 'text-muted-foreground';
		}
	}
</script>

{#if todos.length > 0}
	<div class="rounded-lg border border-border bg-card p-3 shadow-sm">
		<div class="mb-2 flex items-center justify-between gap-3">
			<div class="flex min-w-0 items-center gap-2">
				<ListChecks class="h-4 w-4 shrink-0 text-muted-foreground" />
				<div class="truncate text-sm font-medium">Todo status</div>
			</div>
			<div class="shrink-0 text-xs text-muted-foreground">{completedCount}/{todos.length} done</div>
		</div>

		<div class="space-y-1.5">
			{#each todos as todo, index (`${index}:${todo.content}`)}
				{@const Icon = statusIcon(todo.status)}
				<div class="flex items-start gap-2 rounded-md bg-muted/30 px-2 py-1.5">
					<Icon
						class={`mt-0.5 h-4 w-4 shrink-0 ${statusClass(todo.status)} ${todo.status === 'in_progress' ? 'animate-spin' : ''}`}
					/>
					<div class="min-w-0 flex-1">
						<div class="text-sm leading-5 break-words">{todo.content}</div>
					</div>
				</div>
			{/each}
		</div>
	</div>
{/if}
