<script lang="ts">
	import { Button } from '$lib/components/ui/button';

	interface Props {
		onSave?: () => void;
		onReset?: () => void;
		onClose?: () => void;
	}

	let { onSave, onReset, onClose }: Props = $props();

	function handleSave() {
		onSave?.();
	}

	function handleReset() {
		onReset?.();
	}

	function handleClose() {
		onClose?.();
	}
</script>

<div class="flex justify-between border-t p-6">
	<Button variant="outline" onclick={handleReset}>Reset to default</Button>
	<div class="flex gap-2">
		<Button variant="outline" onclick={handleClose}>Close</Button>
		<Button onclick={handleSave}>Save</Button>
	</div>
</div>
