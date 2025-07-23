<script lang="ts">
	interface Props {
		message?: string;
	}

	let { message = 'Conversations are saved locally in your browser' }: Props = $props();
</script>

<div class="mt-4 border-t pt-4">
	<p class="text-muted-foreground text-center text-xs">
		{message}
	</p>
</div>
