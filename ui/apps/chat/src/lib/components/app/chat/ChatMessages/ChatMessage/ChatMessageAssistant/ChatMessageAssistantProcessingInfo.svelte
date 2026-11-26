<script lang="ts">
	// One status row for a reply that is waiting on the server: loading the
	// model, processing the prompt. It has the same box as a trace row, so
	// the row that comes next takes its place without moving what is below.

	interface Props {
		text: string;
	}

	let { text }: Props = $props();
</script>

<div class="py-1.5 text-body-sm">
	<span class="shimmer-text font-medium">{text}</span>
</div>
