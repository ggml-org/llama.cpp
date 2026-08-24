<script lang="ts">
	import { ChevronRight } from '@lucide/svelte';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';

	interface BreadcrumbItem {
		label: string;
		href?: string;
	}

	interface Props {
		items: BreadcrumbItem[];
		class?: string;
	}

	let { class: className, items }: Props = $props();

	function navigateTo(href: string | undefined) {
		if (href && browser) {
			goto(href);
		}
	}
</script>

<nav class={className} aria-label="Breadcrumb">
	<ol class="flex list-none items-center gap-1.5 p-0 text-sm">
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
					<span class="font-medium text-foreground">{item.label}</span>
				{/if}
			</li>
		{/each}
	</ol>
</nav>
