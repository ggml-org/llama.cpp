<script lang="ts">
	/**
	 * A header component for the chat sidebar that displays the title and a button to create a new conversation.
	 *
	 * @property {() => void} onCreateNew - Called when the "New Conversation" button is clicked.
	 */

	import { Button } from '$lib/components/ui/button';
	import { Plus } from '@lucide/svelte';

	interface Props {
		onCreateNew?: () => void;
	}

	let { onCreateNew }: Props = $props();

	function handleCreateNew() {
		onCreateNew?.();
	}
</script>

<div class="mb-4 flex items-center justify-between">
	<h1 class="text-xl font-semibold">llama.cpp</h1>

	<Button size="sm" onclick={handleCreateNew} class="h-8 w-8 p-0">
		<Plus class="h-4 w-4" />
	</Button>
</div>
