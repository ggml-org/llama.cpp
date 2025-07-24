<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Trash2, Pencil } from '@lucide/svelte';
	import type { Conversation } from '$lib/types/conversation';

	interface Props {
		conversation: Conversation;
		isActive?: boolean;
		onSelect?: (id: string) => void;
		onEdit?: (id: string) => void;
		onDelete?: (id: string) => void;
	}

	let { conversation, isActive = false, onSelect, onEdit, onDelete }: Props = $props();

	function formatLastModified(timestamp: number) {
		const now = Date.now();
		const diff = now - timestamp;
		const minutes = Math.floor(diff / (1000 * 60));
		const hours = Math.floor(diff / (1000 * 60 * 60));
		const days = Math.floor(diff / (1000 * 60 * 60 * 24));

		if (minutes < 1) return 'Just now';
		if (minutes < 60) return `${minutes}m ago`;
		if (hours < 24) return `${hours}h ago`;
		return `${days}d ago`;
	}

	function handleSelect() {
		onSelect?.(conversation.id);
	}

	function handleEdit(event: Event) {
		event.stopPropagation();
		onEdit?.(conversation.id);
	}

	function handleDelete(event: Event) {
		event.stopPropagation();
		onDelete?.(conversation.id);
	}
</script>

<button
	class="hover:bg-accent group flex w-full cursor-pointer items-center justify-between space-x-3 rounded-lg p-3 text-left transition-colors {isActive
		? 'bg-accent text-accent-foreground border-border border'
		: 'border border-transparent'}"
	onclick={handleSelect}
>
	<div class="flex min-w-0 flex-1 items-center space-x-3">
		<div class="min-w-0 flex-1">
			<p class="truncate text-sm font-medium">{conversation.name}</p>

			<div class="mt-2 flex flex-wrap items-center space-x-2 space-y-2">
				<span class="text-muted-foreground w-full text-xs">
					{formatLastModified(conversation.lastModified)}
				</span>
			</div>
		</div>
	</div>

	<div class="flex items-center space-x-1 opacity-0 transition-opacity group-hover:opacity-100">
		<Button size="sm" variant="ghost" class="h-6 w-6 p-0" onclick={handleEdit}>
			<Pencil class="h-3 w-3" />
		</Button>
		<Button
			size="sm"
			variant="ghost"
			class="text-destructive hover:text-destructive h-6 w-6 p-0"
			onclick={handleDelete}
		>
			<Trash2 class="h-3 w-3" />
		</Button>
	</div>
</button>
