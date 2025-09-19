<script lang="ts">
	import { Button } from '$lib/components/ui/button';

	interface Props {
		onReset?: () => void;
		onSave?: () => void;
	}

	let { onReset, onSave }: Props = $props();

	function handleReset() {
		onReset?.();
	}

	function handleSave() {
		onSave?.();
	}
</script>

<div class="flex justify-between border-t border-border/30 p-6">
	<Button variant="outline" onclick={handleReset}>Reset to default</Button>

	<Button onclick={handleSave}>Save settings</Button>
</div>
