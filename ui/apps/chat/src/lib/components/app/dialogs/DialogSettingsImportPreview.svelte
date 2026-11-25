<script lang="ts">
	import * as AlertDialog from '@ggml-org/llama-ui/ui/alert-dialog';
	import { ArrowRight } from '@lucide/svelte';

	export interface SettingsDiffEntry {
		key: string;
		label: string;
		from: SettingsConfigValue;
		to: SettingsConfigValue;
	}

	interface Props {
		open: boolean;
		diff?: SettingsDiffEntry[];
		onConfirm: () => void;
		onCancel: () => void;
	}

	let { diff = [], onCancel, onConfirm, open = $bindable() }: Props = $props();

	function formatValue(value: SettingsConfigValue): string {
		if (value === undefined) return 'default';

		if (typeof value === 'string') {
			return value || '(empty)';
		}

		return String(value);
	}

	function handleOpenChange(newOpen: boolean) {
		if (!newOpen) {
			onCancel();
		}
	}
</script>

<AlertDialog.Root onOpenChange={handleOpenChange} {open}>
	<AlertDialog.Content class="sm:max-w-2xl">
		<AlertDialog.Header>
			<AlertDialog.Title>Import settings?</AlertDialog.Title>
		</AlertDialog.Header>

		<div
			class="max-h-[60vh] overflow-y-auto rounded-md border border-border/30 bg-muted/50 matte-well dark:border-border/20 dark:bg-muted/80"
		>
			{#if diff.length === 0}
				<p class="p-4 text-body-sm text-muted-foreground">No settings would change.</p>
			{:else}
				<div class="divide-y divide-border/30 dark:divide-border/20">
					{#each diff as entry (entry.key)}
						<div class="flex items-center gap-3 p-3">
							<div class="min-w-0 flex-1">
								<p class="truncate text-label">{entry.label}</p>

								<p class="truncate text-caption text-muted-foreground">{entry.key}</p>
							</div>

							<div class="flex shrink-0 items-center gap-2 text-body-sm">
								<span class="line-through text-muted-foreground">{formatValue(entry.from)}</span>

								<ArrowRight class="h-4 w-4 text-muted-foreground" />

								<span class="font-medium">{formatValue(entry.to)}</span>
							</div>
						</div>
					{/each}
				</div>
			{/if}
		</div>

		<AlertDialog.Footer>
			<AlertDialog.Cancel onclick={onCancel}>Cancel</AlertDialog.Cancel>

			<AlertDialog.Action onclick={onConfirm}>Import</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
