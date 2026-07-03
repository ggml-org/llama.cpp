<script lang="ts">
	interface Props {
		isEmpty: boolean;
	}

	let { isEmpty = false }: Props = $props();
</script>

<div
	class={[
		'pointer-events-none mb-4 hidden px-4 text-center text-balance',
		isEmpty && 'mb-[calc(50dvh-8rem)] md:mb-10 pointer-events-auto block!'
	]}
>
	<h1 class="mb-2 text-2xl font-semibold tracking-tight md:text-3xl">Hello there</h1>

	<p class="text-muted-foreground md:text-lg">What can i do for you today?</p>
</div>
