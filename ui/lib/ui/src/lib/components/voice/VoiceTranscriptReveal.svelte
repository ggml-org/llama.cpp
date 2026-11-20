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
	import {
		VOICE_INTERACTIVE_SELECTOR,
		VOICE_NO_HOVER_QUERY,
		VOICE_TRANSCRIPT_LABELS
	} from '../../constants/voice.constants';
	import { PointerType } from '../../enums/ui.enums';
	import { VoiceTranscriptStatus } from '../../enums/voice.enums';

	interface Props {
		/** The clip, usually a VoiceWaveform. */
		children: Snippet;
		transcript?: string | null;
		/** Left out: `done` with a transcript, `idle` without. */
		status?: VoiceTranscriptStatus;
		/** Why there is no transcript, in words for the user, said after the `error` status' own label. */
		errorReason?: string | null;
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
		errorReason = null,
		spoken = null,
		status,
		transcript = null
	}: Props = $props();

	const id = $props.id();

	let root = $state<HTMLDivElement>();
	const initiallyOpen = () => defaultOpen;
	let open = $state(initiallyOpen());
	let noHover = $state(false);
	let hovered = $state(false);
	let keyboardFocus = $state(false);

	let shown = $derived<VoiceTranscriptStatus>(status ?? (transcript ? VoiceTranscriptStatus.DONE : VoiceTranscriptStatus.IDLE));
	let hasContent = $derived(shown !== VoiceTranscriptStatus.IDLE);
	let showToggle = $derived(hasContent && (noHover || open));
	let revealed = $derived(hasContent && (open || hovered || keyboardFocus));

	// plain matchMedia: in the app's type check svelte/reactivity resolves to the library's own copy of Svelte
	$effect(() => {
		const query = window.matchMedia(VOICE_NO_HOVER_QUERY);
		const update = () => (noHover = query.matches);

		update();
		query.addEventListener('change', update);

		return () => query.removeEventListener('change', update);
	});

	function handlePointerEnter(event: PointerEvent) {
		if (event.pointerType === PointerType.MOUSE) hovered = true;
	}

	function handlePointerLeave(event: PointerEvent) {
		if (event.pointerType === PointerType.MOUSE) hovered = false;
	}

	// a tap where there is no hover toggles; a mouse opens it by hovering instead
	function handlePointerUp(event: PointerEvent) {
		if (!hasContent || event.pointerType === PointerType.MOUSE) return;
		if ((event.target as Element | null)?.closest(VOICE_INTERACTIVE_SELECTOR)) return;

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
				aria-label={open ? VOICE_TRANSCRIPT_LABELS.hide : VOICE_TRANSCRIPT_LABELS.show}
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
			aria-busy={shown === VoiceTranscriptStatus.TRANSCRIBING || undefined}
			data-slot="voice-transcript"
		>
			<div class="min-h-0 overflow-hidden">
				<div class="pt-2">
					{#if shown === VoiceTranscriptStatus.TRANSCRIBING}
						<div class="flex flex-col gap-1.5">
							<span class="text-caption text-muted-foreground">{VOICE_TRANSCRIPT_LABELS.transcribing}</span>
							<Skeleton
								class="h-2.5 w-3/4 rounded-full bg-muted-foreground/20 motion-reduce:animate-none"
							/>
						</div>
					{:else if shown === VoiceTranscriptStatus.ERROR}
						<p class="text-caption break-words text-muted-foreground">
							{VOICE_TRANSCRIPT_LABELS.error}{errorReason ? `: ${errorReason}` : ''}
						</p>
					{:else if transcript}
						<p class="text-body-sm break-words whitespace-pre-wrap text-muted-foreground">
							<span class="sr-only">{VOICE_TRANSCRIPT_LABELS.prefix}</span><VoiceSpokenText {spoken} text={transcript} />
						</p>
					{:else}
						<p class="text-caption text-muted-foreground">{VOICE_TRANSCRIPT_LABELS.empty}</p>
					{/if}
				</div>
			</div>
		</div>
	{/if}
</div>
