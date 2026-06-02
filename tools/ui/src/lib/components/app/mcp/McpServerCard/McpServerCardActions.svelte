<script lang="ts">
	import { KeyRound, Trash2, RefreshCw, Pencil } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		isHealthChecking: boolean;
		showAuthorize?: boolean;
		onEdit: () => void;
		onRefresh: () => void;
		onAuthorize?: () => void;
		onDelete: () => void;
	}

	let { isHealthChecking, showAuthorize = false, onEdit, onRefresh, onAuthorize, onDelete }: Props =
		$props();
</script>

<div class="flex shrink-0 items-center gap-1">
	<Button variant="ghost" size="icon" class="h-7 w-7" onclick={onEdit} aria-label="Edit">
		<Pencil class="h-3.5 w-3.5" />
	</Button>

	<Button
		variant="ghost"
		size="icon"
		class="h-7 w-7"
		onclick={onRefresh}
		disabled={isHealthChecking}
		aria-label="Refresh"
	>
		<RefreshCw class="h-3.5 w-3.5" />
	</Button>

	{#if showAuthorize && onAuthorize}
		<Button
			variant="ghost"
			size="sm"
			class="h-7 gap-1.5 px-2 text-xs"
			onclick={onAuthorize}
			disabled={isHealthChecking}
			aria-label="Authorize"
		>
			<KeyRound class="h-3.5 w-3.5" />
			<span>Authorize</span>
		</Button>
	{/if}

	<Button
		variant="ghost"
		size="icon"
		class="hover:text-destructive-foreground h-7 w-7 text-destructive hover:bg-destructive/10"
		onclick={onDelete}
		aria-label="Delete"
	>
		<Trash2 class="h-3.5 w-3.5" />
	</Button>
</div>
