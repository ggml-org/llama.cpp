<script lang="ts">
	import { ChevronRight } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { browser } from '$app/environment';

	interface BreadcrumbItem {
		label: string;
		href?: string;
	}

	interface Props {
		items: BreadcrumbItem[];
		class?: string;
	}

	let { items, class: className }: Props = $props();

	function navigateTo(href: string | undefined) {
		if (href && browser) {
			goto(href);
		}
	}
</script>

<nav class:className>
	<ol class="flex items-center gap-1.5 text-sm">
		{#each items as item, i (item.label)}
			<li class="flex items-center gap-1.5">
				{#if i > 0}
					<ChevronRight class="h-3.5 w-3.5 text-muted-foreground" />
				{/if}

				{#if item.href && i < items.length - 1}
					<a
						href={item.href}
						class="text-muted-foreground transition-colors hover:text-foreground"
						onclick={(e) => {
							e.preventDefault();
							navigateTo(item.href);
						}}
					>
						{item.label}
					</a>
				{:else}
					<span class="text-foreground font-medium">{item.label}</span>
				{/if}
			</li>
		{/each}
	</ol>
</nav>

<style>
	nav {
		display: flex;
		align-items: center;
	}

	ol {
		list-style: none;
		padding: 0;
		margin: 0;
		display: flex;
		align-items: center;
		gap: 0.375rem;
	}

	li {
		display: flex;
		align-items: center;
		gap: 0.375rem;
	}

	a {
		text-decoration: none;
	}
</style>
