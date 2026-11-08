<script lang="ts">
	import { Button } from '$lib/components/ui/button';

	interface Props {
		hasChanges?: boolean;
		onSave?: () => void;
	}

	let { hasChanges = true, onSave }: Props = $props();

	function handleSave() {
		onSave?.();
	}
</script>

{#if hasChanges}
	<div class="sticky bottom-0 mx-auto mt-4 flex w-full justify-end pb-4 md:pb-0">
		<Button onclick={handleSave}>Save settings</Button>
	</div>
{/if}
