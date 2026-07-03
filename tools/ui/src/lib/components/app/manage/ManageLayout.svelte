<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		title: string;
		icon?: Snippet;
		actions?: Snippet;
		children: Snippet;
	}

	let { title, icon, actions, children }: Props = $props();
</script>

<div class="mx-auto w-full p-4 md:p-8 md:py-8">
	<div
		class="sticky top-0 z-10 -mt-4 -ml-4 -mr-4 mb-2 flex items-start gap-4 px-4 md:static md:z-auto md:m-0 md:mb-6 md:justify-between md:px-0"
	>
		<div class="flex min-w-0 items-center gap-2">
			{#if icon}
				{@render icon()}
			{/if}
			<h1 class="truncate text-lg font-semibold md:text-2xl">{title}</h1>
		</div>

		{#if actions}
			<div class="hidden shrink-0 md:flex">
				{@render actions()}
			</div>
		{/if}
	</div>

	{@render children()}

	{#if actions}
		<div class="fixed right-6 bottom-6 z-20 md:hidden">
			{@render actions()}
		</div>
	{/if}
</div>
