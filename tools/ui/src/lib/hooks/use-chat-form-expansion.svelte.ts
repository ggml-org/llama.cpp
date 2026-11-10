import { useKeyboardInset } from './use-keyboard-inset.svelte';
import {
	type CollapsedHeightHoldOptions,
	createCollapsedHeightHold
} from '$lib/components/app/chat/ChatForm/chat-form-collapsed-height';
import {
	SHEET_AT_REST,
	type SheetFrame,
	sheetFrameAt,
	type SheetRects,
	shiftSheetRects,
	shouldExpandChatForm
} from '$lib/components/app/chat/ChatForm/chat-form-expand';
import { CHAT_FORM_EASE_OUT, CHAT_FORM_SHEET_MS, UI_DATA_ATTRS } from '$lib/constants';
import { deviceStore, uiStore } from '$lib/stores';
import { flushSync, onDestroy, tick, untrack } from 'svelte';

export interface UseChatFormExpansionOptions extends CollapsedHeightHoldOptions {
	/** Whether this form expands at all. */
	isEnabled: () => boolean;
	/** What the message belongs to (the conversation): another one starts collapsed. */
	getResetKey: () => unknown;
	/** The box around the form's actions. */
	getActions: () => HTMLElement | undefined;
	focusInput: () => void;
}

/**
 * The chat form's expanded editor: entered by hand, or on a desktop once the message being
 * written asks for it (see shouldExpandChatForm), kept while it has text, left once it is
 * cleared or sent. Collapsing by hand holds until then too. On a phone it is a full-screen
 * sheet that slides from and back to the card (see sheetFrameAt).
 */
export function useChatFormExpansion(options: UseChatFormExpansionOptions) {
	let isExpanded = $state(false);
	let isCollapsedByUser = false;
	let hadText = false;
	let wasExpanded = false;
	let isSheetMoving = $state(false);
	// WebKit builds the card's backdrop-filter layer with the animation's last clip when
	// the blur comes back as the card lands: it waits for the move to be over
	let isBlurHeld = $state(false);
	let collapsedRects: SheetRects | null = null;
	// iOS Safari resizes its viewport while the sheet is up: the collapsed card's place
	// is held relative to the sheet's bottom edge
	let sheetBottomAtOpen = 0;

	$effect(() => {
		if (!options.isEnabled()) return;

		const text = options.getValue() ?? '';

		untrack(() => {
			if (!text.trim()) {
				// cleared or sent
				if (hadText) reset();

				hadText = false;

				return;
			}

			hadText = true;
		});
	});

	let resetKey = untrack(options.getResetKey);

	$effect(() => {
		const key = options.getResetKey();

		if (key === resetKey) return;

		resetKey = key;
		untrack(reset);
	});

	/** Straight back to the small form, without the sheet's slide, free to expand again. */
	function reset() {
		isCollapsedByUser = false;

		if (isSheetMoving) clearSheetAnimations();

		isExpanded = false;
	}

	/**
	 * Expands for what the user writes or pastes, not for a draft restored on opening a
	 * chat. A phone only expands by hand: the sheet switching in mid-typing moves the text
	 * under the keyboard.
	 */
	function handleInput() {
		if (!options.isEnabled() || isExpanded || isCollapsedByUser || deviceStore.isMobile) return;

		isExpanded = shouldExpandChatForm(options.getValue() ?? '');
	}

	$effect(() => {
		if (options.isEnabled()) uiStore.isComposerExpanded = isExpanded;
	});

	// the phone sheet fills the screen, so its actions would sit under an iOS keyboard
	useKeyboardInset(() => isExpanded && deviceStore.isMobile);

	// the page around the phone sheet takes its fill (see app.css) once it has settled:
	// repainting the whole page under it mid-move costs frames
	$effect(() => {
		if (!options.isEnabled() || !isExpanded || isSheetMoving) return;

		document.documentElement.setAttribute(UI_DATA_ATTRS.CHAT_FORM_SHEET, '');

		return () => document.documentElement.removeAttribute(UI_DATA_ATTRS.CHAT_FORM_SHEET);
	});

	function animatesSheet(): boolean {
		return deviceStore.isMobile && !matchMedia('(prefers-reduced-motion: reduce)').matches;
	}

	function sheetBottom(): number {
		return options.getCard()?.closest('form')?.getBoundingClientRect().bottom ?? 0;
	}

	function measure(): SheetRects | null {
		const card = options.getCard();
		const actions = options.getActions();

		if (!card || !actions) return null;

		const { height, left, right, top } = card.getBoundingClientRect();

		return {
			actionsTop: actions.getBoundingClientRect().top,
			card: { height, left, right, top },
			padTop: parseFloat(getComputedStyle(card).paddingTop) || 0
		};
	}

	/**
	 * Plays the frames on the card and its actions, none held (WebKit keeps drawing a held
	 * clip after the layout changed). `onEnd` switches the layout as a collapse ends, before
	 * that frame is painted.
	 */
	function animateSheet(
		frames: SheetFrame[],
		duration = CHAT_FORM_SHEET_MS,
		onEnd?: () => void
	): Promise<unknown> {
		const card = options.getCard();
		const actions = options.getActions();

		// the form went away mid-move
		if (!card || !actions) return Promise.resolve();

		const keyframeOptions = { duration, easing: CHAT_FORM_EASE_OUT };
		const animations = [
			card.animate(
				frames.map((frame) => frame.card),
				keyframeOptions
			),
			actions.animate(
				frames.map((frame) => frame.actions),
				keyframeOptions
			)
		];

		if (onEnd) animations[0].addEventListener('finish', onEnd, { once: true });

		return Promise.all(animations.map((animation) => animation.finished));
	}

	function clearSheetAnimations() {
		for (const el of [options.getCard(), options.getActions()]) {
			el?.getAnimations()
				.filter((animation) => !(animation instanceof CSSTransition))
				.forEach((animation) => animation.cancel());
		}
	}

	async function expandSheet(from: SheetRects) {
		isSheetMoving = true;
		// measure once the page has laid the form out flush with the screen, before a paint
		await tick();
		sheetBottomAtOpen = sheetBottom();

		try {
			const sheet = measure();

			if (sheet) await animateSheet([sheetFrameAt(sheet, from), SHEET_AT_REST]);
		} catch {
			// cancelled: the form went away or collapsed
		}

		clearSheetAnimations();
		isSheetMoving = false;
	}

	async function collapseSheet() {
		const sheet = measure();

		if (!collapsedRects || !sheet) {
			isExpanded = false;

			return;
		}

		const target = shiftSheetRects(collapsedRects, sheetBottom() - sheetBottomAtOpen);

		isSheetMoving = true;
		isBlurHeld = true;

		try {
			await animateSheet([SHEET_AT_REST, sheetFrameAt(sheet, target)], CHAT_FORM_SHEET_MS, () => {
				flushSync(() => (isExpanded = false));
			});
		} catch {
			// cancelled: settle at once
		}

		isExpanded = false;
		await tick();

		// the text may have changed while expanded, and with it the collapsed height
		const settled = measure();

		if (settled && Math.abs(settled.card.top - target.card.top) > 1) {
			const dy = target.card.top - settled.card.top;

			await animateSheet(
				[
					{ actions: {}, card: { top: `${dy}px` } },
					{ actions: {}, card: { top: '0px' } }
				],
				CHAT_FORM_SHEET_MS / 2
			).catch(() => {});
			clearSheetAnimations();
		}

		isSheetMoving = false;

		await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
		isBlurHeld = false;
	}

	const collapsedHeight = createCollapsedHeightHold(options);

	// before the DOM switches: where the collapsed card is, and how tall
	$effect.pre(() => {
		const expanded = isExpanded;

		untrack(() => {
			if (expanded === wasExpanded) return;

			if (!expanded) {
				collapsedHeight.settle();

				return;
			}

			if (!isSheetMoving && animatesSheet()) collapsedRects = measure();

			collapsedHeight.start();
		});
	});

	// after it switched: ease the sheet in from there
	$effect(() => {
		const expanded = isExpanded;

		untrack(() => {
			if (expanded && !wasExpanded && collapsedRects && !isSheetMoving) {
				void expandSheet(collapsedRects);
			}

			if (!expanded) collapsedRects = null;

			wasExpanded = expanded;
		});
	});

	onDestroy(() => {
		collapsedHeight.destroy();

		if (options.isEnabled()) uiStore.isComposerExpanded = false;
	});

	function toggle() {
		if (isSheetMoving) return;

		// collapsing an empty form leaves the next message free to expand it again
		isCollapsedByUser = isExpanded && !!(options.getValue() ?? '').trim();

		if (isExpanded && animatesSheet()) {
			void collapseSheet();

			return;
		}

		isExpanded = !isExpanded;
		// after the switch, so the editor redrawing as it takes focus does not reach into
		// the heights measured for it
		options.focusInput();
	}

	return {
		handleInput,
		get isBlurHeld() {
			return isBlurHeld;
		},
		get isExpanded() {
			return isExpanded;
		},
		get isSheetMoving() {
			return isSheetMoving;
		},
		toggle
	};
}
