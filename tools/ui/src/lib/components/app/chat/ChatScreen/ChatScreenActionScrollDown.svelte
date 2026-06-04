<script lang="ts">
	import { ArrowDown } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		container: HTMLDivElement | undefined;
		hasProcessingInfoVisible: boolean;
	}

	let { container, hasProcessingInfoVisible }: Props = $props();

	let show = $state(false);

	let buttonBottom = $derived(hasProcessingInfoVisible ? '2rem' : '0');

	function checkVisibility() {
		if (!container) return;
		const { scrollTop, scrollHeight, clientHeight } = container;
		const distanceFromBottom = scrollHeight - clientHeight - scrollTop;
		show = distanceFromBottom > clientHeight * 0.5;
	}

	function scrollToBottom() {
		if (container) {
			container.scrollTo({
				top: container.scrollHeight,
				behavior: 'smooth'
			});
		}
	}

	$effect(() => {
		const c = container;
		if (c) {
			c.addEventListener('scroll', checkVisibility);
			checkVisibility();
			return () => {
				c.removeEventListener('scroll', checkVisibility);
			};
		}
	});
</script>

<div class="relative z-50 mx-auto mb-4 flex max-w-[48rem] justify-center">
	<Button
		onclick={scrollToBottom}
		variant="secondary"
		size="icon"
		disabled={!show}
		class="pointer-events-auto absolute h-10 w-10 rounded-full bg-background/80 shadow-lg backdrop-blur-sm transition-all duration-200 hover:bg-muted/80"
		style="bottom: {buttonBottom}; transform: translateY({show ? '0' : '2rem'}); opacity: {show
			? 1
			: 0};"
		aria-label="Scroll to bottom"
	>
		<ArrowDown class="h-4 w-4" />
	</Button>
</div>
