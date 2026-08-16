<script lang="ts">
	import { X } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app/actions';
	import type { Component, Snippet } from 'svelte';
	import { fade } from 'svelte/transition';

	interface Props {
		icon: Component;
		title: string;
		onClose: () => void;
		children: Snippet;
		class?: string;
		headerClass?: string;
	}

	let {
		children,
		class: className = '',
		headerClass = '',
		icon: Icon,
		onClose,
		title
	}: Props = $props();
</script>

<div
	data-testid="standalone-page-shell"
	in:fade={{ duration: 150 }}
	class="flex min-h-[calc(100dvh-4rem)] flex-col {className}"
>
	<div class="fixed top-4.5 right-4 z-50 md:hidden">
		<ActionIcon icon={X} tooltip="Close" onclick={onClose} />
	</div>
	<div
		data-testid="standalone-page-shell-header"
		class="sticky top-0 z-10 mt-4 mb-2 flex items-start gap-4 p-0 px-4 md:justify-between md:p-4 md:px-8 {headerClass}"
	>
		<div class="flex items-center gap-2">
			<Icon class="h-5 w-5 md:h-6 md:w-6" />

			<h1 class="text-lg font-semibold md:text-2xl">{title}</h1>
		</div>
	</div>
	{@render children()}
</div>
