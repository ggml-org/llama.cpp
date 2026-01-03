<!-- [AI] Tool execution status indicator component -->
<script lang="ts">
	import { Loader2, CheckCircle2, XCircle, Wrench } from '@lucide/svelte';
	import { fade } from 'svelte/transition';

	interface Props {
		toolName: string;
		status: 'pending' | 'executing' | 'completed' | 'error';
		result?: string;
		error?: string;
	}

	let { toolName, status, result, error }: Props = $props();

	const statusConfig = {
		pending: {
			icon: Wrench,
			color: 'text-muted-foreground',
			bgColor: 'bg-muted-foreground/10',
			label: 'Pending'
		},
		executing: {
			icon: Loader2,
			color: 'text-blue-500',
			bgColor: 'bg-blue-500/10',
			label: 'Executing'
		},
		completed: {
			icon: CheckCircle2,
			color: 'text-green-500',
			bgColor: 'bg-green-500/10',
			label: 'Completed'
		},
		error: {
			icon: XCircle,
			color: 'text-red-500',
			bgColor: 'bg-red-500/10',
			label: 'Error'
		}
	};

	const config = $derived(statusConfig[status]);
	const Icon = $derived(config.icon);
</script>

<div
	class="inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm {config.bgColor}"
	transition:fade
>
	<svelte:component
		this={Icon}
		class="h-4 w-4 {config.color} {status === 'executing' ? 'animate-spin' : ''}"
	/>
	<span class="font-mono">{toolName}</span>
	<span class="text-xs {config.color}">{config.label}</span>

	{#if error}
		<span class="text-xs text-red-500">- {error}</span>
	{/if}
</div>
