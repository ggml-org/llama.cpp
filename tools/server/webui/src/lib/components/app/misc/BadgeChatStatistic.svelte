<script lang="ts">
	import { BadgeInfo } from '$lib/components/app';
	import { copyToClipboard } from '$lib/utils';
	import type { Component } from 'svelte';

	interface Props {
		class?: string;
		icon: Component;
		value: string | number;
	}

	let { class: className = '', icon: Icon, value }: Props = $props();

	function handleClick() {
		void copyToClipboard(String(value));
	}
</script>

<BadgeInfo class={className} onclick={handleClick}>
	{#snippet icon()}
		<Icon class="h-3 w-3" />
	{/snippet}

	{value}
</BadgeInfo>
