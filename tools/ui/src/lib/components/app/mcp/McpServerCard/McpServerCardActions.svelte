<script lang="ts">
	import { Pencil, RefreshCw, Trash2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		isHealthChecking: boolean;
		onEdit: () => void;
		onRefresh: () => void;
		onDelete: () => void;
	}

	let { isHealthChecking, onDelete, onEdit, onRefresh }: Props = $props();
</script>

<div class="flex shrink-0 items-center gap-1">
	<Button aria-label="Edit" class="h-7 w-7" onclick={onEdit} size="icon" variant="ghost">
		<Pencil class="h-3.5 w-3.5" />
	</Button>

	<Button
		aria-label="Refresh"
		class="h-7 w-7"
		disabled={isHealthChecking}
		onclick={onRefresh}
		size="icon"
		variant="ghost"
	>
		<RefreshCw class="h-3.5 w-3.5" />
	</Button>

	<Button
		aria-label="Delete"
		class="hover:text-destructive-foreground h-7 w-7 text-destructive hover:bg-destructive/10"
		onclick={onDelete}
		size="icon"
		variant="ghost"
	>
		<Trash2 class="h-3.5 w-3.5" />
	</Button>
</div>
