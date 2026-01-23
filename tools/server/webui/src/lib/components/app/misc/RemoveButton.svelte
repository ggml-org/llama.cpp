<script lang="ts">
	import { X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { t } from '$lib/i18n';

	interface Props {
		id: string;
		onRemove?: (id: string) => void;
		class?: string;
	}

	let { id, onRemove, class: className = '' }: Props = $props();
</script>

<Button
	type="button"
	variant="ghost"
	size="sm"
	class="h-6 w-6 bg-white/20 p-0 hover:bg-white/30 {className}"
	onclick={(e) => {
		e.stopPropagation();
		onRemove?.(id);
	}}
	aria-label={t('chat.attachments.remove_file')}
>
	<X class="h-3 w-3" />
</Button>
