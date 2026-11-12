<!--
	One variation at a desktop width and at a phone's, side by side (stacked on a narrow screen). Each frame is a size
	container, so a variation lays itself out with container queries (@md:, @3xl:) and the phone frame shows the phone
	layout without resizing the window.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	let { children }: { children: Snippet } = $props();
</script>

<div class="flex min-h-screen flex-col gap-6 bg-background p-6 xl:flex-row xl:items-start">
	<section class="min-w-0 flex-1">
		<p class="mb-3 font-mono text-label-sm tracking-[0.12em] text-muted-foreground uppercase">Desktop</p>

		<div class="@container overflow-hidden rounded-card bg-background shadow-[0_0_0_1px_var(--line-default)]">
			{@render children()}
		</div>
	</section>

	<section class="w-full shrink-0 xl:w-[375px]">
		<p class="mb-3 font-mono text-label-sm tracking-[0.12em] text-muted-foreground uppercase">Phone · 375</p>

		<div
			class="@container mx-auto w-[375px] overflow-hidden rounded-[2rem] bg-background shadow-[0_0_0_1px_var(--line-default)]"
		>
			{@render children()}
		</div>
	</section>
</div>
