<script lang="ts">
	import { ArrowDown } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	let { container }: { container: HTMLDivElement | undefined } = $props();

	let show = $state(false);

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

<div class="pointer-events-auto relative z-50 mx-auto mb-4 flex max-w-[48rem] justify-center">
	<Button
		onclick={scrollToBottom}
		variant="secondary"
		size="icon"
		class="h-10 w-10 rounded-full bg-background/80 shadow-lg backdrop-blur-sm transition-all duration-200 hover:bg-muted/80"
		aria-label="Scroll to bottom"
		style="transform: translateY({show ? '0' : '20px'}); opacity: {show ? 1 : 0};"
	>
		<ArrowDown class="h-4 w-4" />
	</Button>
</div>
