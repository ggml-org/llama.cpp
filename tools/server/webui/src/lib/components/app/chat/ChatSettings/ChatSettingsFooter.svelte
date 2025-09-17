<script lang="ts">
	import { Button } from '$lib/components/ui/button';

	interface Props {
		onClose?: () => void;
		onReset?: () => void;
		onSave?: () => void;
	}

	let { onClose, onReset, onSave }: Props = $props();

	function handleClose() {
		onClose?.();
	}

	function handleReset() {
		onReset?.();
	}

	function handleSave() {
		onSave?.();
	}
</script>

<div class="flex justify-between border-t border-border/30 p-6">
	<Button variant="outline" onclick={handleReset}>Reset to default</Button>

	<div class="flex gap-2">
		<Button variant="outline" onclick={handleClose}>Close</Button>

		<Button onclick={handleSave}>Save</Button>
	</div>
</div>
