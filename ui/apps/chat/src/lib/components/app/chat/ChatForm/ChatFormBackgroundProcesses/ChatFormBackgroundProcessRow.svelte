<script lang="ts">
	// One background process in the dialog: its description and a Stop action, above the
	// same terminal block as a shell tool call, collapsed, with the live log tail inside.
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { Square, SquareTerminal, Trash2 } from '@lucide/svelte';
	import { CollapsibleTerminalBlock } from '#lib/components/app/index.ts';
	import { BACKGROUND_PROCESS_LOG_POLL_MS, MS_PER_SECOND } from '#lib/constants/index.ts';
	import { ToolsService } from '#lib/services/tools.service.ts';
	import { backgroundProcessesStore } from '#lib/stores/index.ts';
	import type { BackgroundProcess } from '#lib/types/index.ts';
	import { formatRelativeTime } from '#lib/utils/index.ts';

	interface Props {
		process: BackgroundProcess;
		open?: boolean;
	}

	let { open = $bindable(false), process }: Props = $props();

	let log = $state('');
	let stopping = $state(false);
	let logEl: HTMLPreElement | undefined = $state();

	// the log is read only while the block is open, and kept pinned to its end
	$effect(() => {
		if (!open) return;

		const controller = new AbortController();
		const read = async () => {
			try {
				log = await ToolsService.getBackgroundProcessLog(process.id, controller.signal);
				requestAnimationFrame(() => {
					if (logEl) logEl.scrollTop = logEl.scrollHeight;
				});
			} catch {
				// the process left the list: the next refresh drops this row
			}
		};

		void read();
		const timer = setInterval(read, BACKGROUND_PROCESS_LOG_POLL_MS);

		return () => {
			clearInterval(timer);
			controller.abort();
		};
	});

	async function stop(event: MouseEvent) {
		event.stopPropagation();
		stopping = true;

		try {
			await backgroundProcessesStore.stop(process.id);
		} finally {
			stopping = false;
		}
	}
</script>

<div class="grid gap-2" data-background-process={process.id}>
	<div class="flex items-center gap-2">
		<SquareTerminal
			class={[
				'size-icon-md shrink-0',
				process.running ? 'text-fg-success' : 'text-muted-foreground'
			]}
		/>

		<span class="min-w-0 flex-1 truncate text-body-sm font-medium">
			{process.description || process.command}
		</span>

		<Button
			aria-label={process.running ? 'Stop process' : 'Remove process'}
			class="h-auto gap-1.5 px-2 py-1 text-caption"
			disabled={stopping}
			onclick={stop}
			size="sm"
			variant="danger-secondary"
		>
			{#if process.running}
				<Square class="size-icon-xs" />
			{:else}
				<Trash2 class="size-icon-xs" />
			{/if}

			{process.running ? 'Stop' : 'Remove'}
		</Button>
	</div>

	<CollapsibleTerminalBlock bind:open icon={SquareTerminal}>
		{#snippet titleSnippet()}
			<span class="min-w-0 truncate font-mono text-muted-foreground">{process.command}</span>
		{/snippet}

		<div class="grid gap-2">
			<div class="font-mono text-caption text-muted-foreground">
				{process.running ? 'running' : 'exited'} · pid {process.pid} · started
				{formatRelativeTime(process.started_at * MS_PER_SECOND)}
			</div>

			<pre
				bind:this={logEl}
				class="max-h-80 overflow-auto font-mono text-caption leading-relaxed whitespace-pre-wrap">{log ||
					'No output yet'}</pre>
		</div>
	</CollapsibleTerminalBlock>
</div>
