<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		text: string;
		class?: string;
	}

	let { text, class: className = '' }: Props = $props();

	let textElement: HTMLSpanElement | undefined = $state();
	let isTruncated = $state(false);

	function checkTruncation() {
		if (textElement) {
			isTruncated = textElement.scrollWidth > textElement.clientWidth;
		}
	}

	$effect(() => {
		if (textElement) {
			checkTruncation();

			const observer = new ResizeObserver(checkTruncation);
			observer.observe(textElement);

			return () => observer.disconnect();
		}
	});
</script>

{#if isTruncated}
	<Tooltip.Root>
		<Tooltip.Trigger class={className}>
			<span bind:this={textElement} class="block truncate">
				{text}
			</span>
		</Tooltip.Trigger>

		<Tooltip.Content class="z-[9999]">
			<p>{text}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{:else}
	<span bind:this={textElement} class="{className} block truncate">
		{text}
	</span>
{/if}
