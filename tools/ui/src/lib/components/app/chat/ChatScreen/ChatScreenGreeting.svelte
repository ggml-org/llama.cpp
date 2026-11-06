<script lang="ts">
	import { CHAT_GREETING_ENTER_DELAY_MS } from '$lib/constants';
	import { serverStore } from '$lib/stores';

	interface Props {
		isEmpty: boolean;
	}

	let { isEmpty = false }: Props = $props();

	let hasEntered = $state(false);

	// Starts the entry only once the greeting has been painted hidden, so it
	// plays on a refresh or a first visit too, when the first frames are busy
	// loading the app, not just after the form moves.
	$effect(() => {
		if (!isEmpty) {
			hasEntered = false;

			return;
		}

		let frame = requestAnimationFrame(() => {
			frame = requestAnimationFrame(() => (hasEntered = true));
		});

		return () => cancelAnimationFrame(frame);
	});

	let enterClass = $derived([
		'transition-opacity duration-300 ease-[cubic-bezier(0.23,1,0.32,1)]',
		!hasEntered && 'opacity-0'
	]);
</script>

<div
	class={[
		'pointer-events-none mb-4 hidden px-4 text-center text-balance',
		isEmpty && 'mb-[calc(50dvh-8rem)] md:mb-8 pointer-events-auto block!',
		// out of the page flow on desktop, just above the form: inside the form
		// wrapper's top padding, so showing it never changes the wrapper's height
		isEmpty && 'md:absolute md:inset-x-0 md:bottom-[calc(100%-var(--chat-form-padding-top))]'
	]}
>
	<div style:transition-delay="{CHAT_GREETING_ENTER_DELAY_MS}ms" class={enterClass}>
		<h1 class="mb-2 text-2xl font-semibold tracking-tight md:text-3xl">Hello there</h1>

		<p class="text-muted-foreground md:text-lg">
			{serverStore.props?.modalities?.audio ? 'Record audio, type a message ' : 'Type a message'} or upload
			files to get started
		</p>
	</div>
</div>
