<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		text: string;
		class?: string;
		showTooltip?: boolean;
	}

	let { class: className = '', showTooltip = true, text }: Props = $props();

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

{#if isTruncated && showTooltip}
	<Tooltip.Root>
		<!-- as wide as the plain span below: a trigger that shrank to its text would stop
		     truncating, flip back to the plain span and loop -->
		<Tooltip.Trigger class="{className} block w-full min-w-0 text-left">
			<span bind:this={textElement} class="block truncate">
				{text}
			</span>
		</Tooltip.Trigger>

		<Tooltip.Content class="z-[9999]">
			<p>{text}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{:else}
	<span bind:this={textElement} class="{className} block min-w-0 truncate">
		{text}
	</span>
{/if}
