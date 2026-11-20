<!--
	A voice clip with its transcript under it, open from the start with `defaultOpen` (a toggle key beside the clip folds
	it). Otherwise only the clip (the `children`, usually a VoiceWaveform) shows at first; the transcript opens below on
	hover and on keyboard focus inside, or with a tap where there is no hover (the toggle key, or a tap on the clip
	outside its controls). While it is transcribing, the open transcript shows that it is on its way.
	The transcript stays in the DOM while closed, so screen readers always reach it.
-->
<script lang="ts">
	import { ChevronDown } from '@lucide/svelte';
	import type { Snippet } from 'svelte';
	import { Button } from '../ui/button';
	import { Skeleton } from '../ui/skeleton';
	import VoiceSpokenText from './VoiceSpokenText.svelte';
	import { cn } from '../ui/utils';
	import type { VoiceTranscriptStatus } from '../../voice/types';

	interface Props {
		/** The clip, usually a VoiceWaveform. */
		children: Snippet;
		transcript?: string | null;
		/** Left out: `done` with a transcript, `idle` without. */
		status?: VoiceTranscriptStatus;
		/** Start with the transcript open; read once, the toggle key changes it after that. */
		defaultOpen?: boolean;
		/** Characters of the transcript spoken so far, to follow playback; left out, all of it reads as spoken. */
		spoken?: number | null;
		class?: string;
	}

	let {
		children,
		class: className,
		defaultOpen = false,
		spoken = null,
		status,
		transcript = null
	}: Props = $props();

	const INTERACTIVE = 'button, a, input, textarea, select, [role="slider"], [tabindex]';
	const id = $props.id();

	let root = $state<HTMLDivElement>();
	const initiallyOpen = () => defaultOpen;
	let open = $state(initiallyOpen());
	let noHover = $state(false);
	let hovered = $state(false);
	let keyboardFocus = $state(false);

	let shown = $derived<VoiceTranscriptStatus>(status ?? (transcript ? 'done' : 'idle'));
	let hasContent = $derived(shown !== 'idle');
	let showToggle = $derived(hasContent && (noHover || open));
	let revealed = $derived(hasContent && (open || hovered || keyboardFocus));

	// plain matchMedia: in the app's type check svelte/reactivity resolves to the library's own copy of Svelte
	$effect(() => {
		const query = window.matchMedia('(hover: none)');
		const update = () => (noHover = query.matches);

		update();
		query.addEventListener('change', update);

		return () => query.removeEventListener('change', update);
	});

	function handlePointerEnter(event: PointerEvent) {
		if (event.pointerType === 'mouse') hovered = true;
	}

	function handlePointerLeave(event: PointerEvent) {
		if (event.pointerType === 'mouse') hovered = false;
	}

	// a tap where there is no hover toggles; a mouse opens it by hovering instead
	function handlePointerUp(event: PointerEvent) {
		if (!hasContent || event.pointerType === 'mouse') return;
		if ((event.target as Element | null)?.closest(INTERACTIVE)) return;

		open = !open;
	}

	// only keyboard focus opens it: after a click the mouse already hovers, and a tap toggles
	function handleFocusIn(event: FocusEvent) {
		keyboardFocus = (event.target as Element | null)?.matches(':focus-visible') ?? false;
	}

	function handleFocusOut(event: FocusEvent) {
		if (!root?.contains(event.relatedTarget as Node | null)) keyboardFocus = false;
	}
</script>

<!-- the tap is a shortcut: the toggle key does the same for keyboards and screen readers -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	bind:this={root}
	class={cn('flex min-w-0 flex-col', className)}
	data-revealed={revealed || undefined}
	data-slot="voice-transcript-reveal"
	data-status={shown}
	onfocusin={handleFocusIn}
	onfocusout={handleFocusOut}
	onpointerenter={handlePointerEnter}
	onpointerleave={handlePointerLeave}
	onpointerup={handlePointerUp}
>
	<div class="flex min-w-0 items-center gap-1">
		<div class="min-w-0 flex-1">
			{@render children()}
		</div>

		{#if showToggle}
			<Button
				class="size-8 shrink-0 rounded-full p-0 text-muted-foreground"
				aria-controls={id}
				aria-expanded={open}
				aria-label={open ? 'Hide transcript' : 'Show transcript'}
				onclick={() => (open = !open)}
				variant="ghost"
			>
				<ChevronDown
					class={cn(
						'size-icon-md transition-transform duration-200 ease-out motion-reduce:transition-none',
						open && 'rotate-180'
					)}
				/>
			</Button>
		{/if}
	</div>

	{#if hasContent}
		<div
			{id}
			class={[
				'grid transition-[grid-template-rows,opacity] duration-200 ease-out motion-reduce:transition-none',
				revealed ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'
			]}
			aria-busy={shown === 'transcribing' || undefined}
			data-slot="voice-transcript"
		>
			<div class="min-h-0 overflow-hidden">
				<div class="pt-2">
					{#if shown === 'transcribing'}
						<div class="flex flex-col gap-1.5">
							<span class="text-caption text-muted-foreground">Transcribing...</span>
							<Skeleton
								class="h-2.5 w-3/4 rounded-full bg-muted-foreground/20 motion-reduce:animate-none"
							/>
						</div>
					{:else if shown === 'error'}
						<p class="text-caption text-muted-foreground">Transcription unavailable</p>
					{:else if transcript}
						<p class="text-body-sm break-words whitespace-pre-wrap text-muted-foreground">
							<span class="sr-only">Transcript: </span><VoiceSpokenText {spoken} text={transcript} />
						</p>
					{:else}
						<p class="text-caption text-muted-foreground">No speech detected</p>
					{/if}
				</div>
			</div>
		</div>
	{/if}
</div>
