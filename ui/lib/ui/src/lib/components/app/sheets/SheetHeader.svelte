<!--
	A sheet's header: a round key in the top-left corner (close on the first page, back on the pages
	pushed over it) and the title centered on that key. It goes inside a Drawer's content, which
	takes SHEET_CONTENT_CLASS.
-->
<script lang="ts">
	import { ChevronLeft, X } from '@lucide/svelte';
	import * as Drawer from '../../ui/drawer';
	import { cn } from '../../ui/utils';
	import { ICON_CLASS_DEFAULT, SHEET_TEXT } from '../../../constants';

	interface Props {
		backLabel?: string;
		/** Show back in place of close: a page sits over the first one. */
		canGoBack?: boolean;
		class?: string;
		closeLabel?: string;
		/** Lay the header over the content, which scrolls under its blurred, fading surface. */
		floating?: boolean;
		onBack?: () => void;
		title: string;
	}

	let {
		backLabel = SHEET_TEXT.BACK,
		canGoBack = false,
		class: className,
		closeLabel = SHEET_TEXT.CLOSE,
		floating = false,
		onBack,
		title
	}: Props = $props();
</script>

<header
	class={cn(
		'grid grid-cols-[3rem_1fr_3rem] items-center px-3 pb-7',
		floating
			? 'pointer-events-none absolute inset-x-0 top-0 isolate z-10 pt-5.5 *:pointer-events-auto'
			: 'shrink-0 pt-2.5',
		className
	)}
>
	{#if floating}
		<div
			aria-hidden="true"
			class="pointer-events-none! absolute inset-0 -z-10 bg-(--float-panel-background) backdrop-blur-xl mask-[linear-gradient(to_bottom,black_60%,transparent)]"
		></div>
	{/if}

	{#if canGoBack}
		<button aria-label={backLabel} class="sheet-key" onclick={onBack} type="button">
			<ChevronLeft class="size-5" />
		</button>
	{:else}
		<Drawer.Close aria-label={closeLabel} class="sheet-key">
			<X class={ICON_CLASS_DEFAULT} />
		</Drawer.Close>
	{/if}

	<Drawer.Title class="col-start-2 truncate text-center text-body font-semibold">
		{title}
	</Drawer.Title>
</header>
