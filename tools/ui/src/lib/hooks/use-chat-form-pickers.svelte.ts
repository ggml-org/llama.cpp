import { getChatCommands, PROMPT_TRIGGER_PREFIX } from '$lib/constants';
import { ChatFormCommandAction, KeyboardKey } from '$lib/enums';
import type { ChatFormCommand } from '$lib/types';
import {
	findCommandToken,
	findMentionToken,
	takeCommandDismissSnapshot,
	takeMentionDismissSnapshot,
	type CommandDismissSnapshot,
	type MentionDismissSnapshot
} from '$lib/utils';

/**
 * Dependencies injected as getters so the hook stays free of component
 * state and store circular imports.
 */
export interface UseChatFormPickersOptions {
	getValue: () => string;
	/** Also fires the form's onChange. */
	setValue: (value: string) => void;
	/** Undefined when unmounted. */
	getCaretOffset: () => number | undefined;
	setCaretOffset: (offset: number) => void;
	focusInput: () => void;
	/** Gates `/model`. */
	getShowModelSelector: () => boolean;
	/** Gates `/prompt`. */
	hasPrompts: () => boolean;
	/** Gates `/cwd`. */
	hasBuiltinTools: () => boolean;
	getCwd: () => string | null;
	/** Mention search fallback scope. */
	getServerHome: () => string | null;
	openModelSelector: () => void;
	/** Delegate a keydown to the mounted pickers component, if any. */
	getPickersRef: () => { handleKeydown(event: KeyboardEvent): boolean } | undefined;
}

/**
 * Chat-form picker state and the `/`+`@` routing that drives them.
 * Owns open/query state, dismiss snapshots and slash-command dispatch;
 * textarea/caret/attachment handling stays in the chat form.
 */
export function useChatFormPickers(opts: UseChatFormPickersOptions) {
	let isCommandPickerOpen = $state(false);
	let commandQuery = $state('');
	let isPromptPickerOpen = $state(false);
	let promptSearchQuery = $state('');
	let isMentionPickerOpen = $state(false);
	let mentionQuery = $state('');
	let isWorkingDirectoryPickerOpen = $state(false);
	let workingDirectoryQuery = $state('');

	/**
	 * Last dismissed `@`-mention token; while intact, the picker does not
	 * reopen, so an escaped `@<query>` stays literal until edited.
	 */
	let mentionDismissedSnapshot: MentionDismissSnapshot | null = null;

	/**
	 * Last dismissed `/`-command token; while intact, the picker neither
	 * reopens nor instant-dispatches, so an escaped `/name` stays literal.
	 */
	let commandDismissedSnapshot: CommandDismissSnapshot | null = null;

	// Scope the @-mention search to the picked cwd, falling back to the
	// server home so the picker still finds matches before a directory is set.
	const mentionScopePath = $derived(opts.getCwd() ?? opts.getServerHome() ?? null);

	const availableCommands = $derived(
		getChatCommands({
			showModelSelector: opts.getShowModelSelector(),
			hasPrompts: opts.hasPrompts,
			hasBuiltinTools: opts.hasBuiltinTools
		})
	);

	/**
	 * Dispatch a selected slash command: consume the token and open the
	 * target picker. `args` seeds the target search where applicable.
	 */
	function dispatchCommand(command: ChatFormCommand, args: string) {
		switch (command.action) {
			case ChatFormCommandAction.PROMPT:
				isWorkingDirectoryPickerOpen = false;
				opts.setValue('');
				isPromptPickerOpen = true;
				promptSearchQuery = args.trim();
				break;
			case ChatFormCommandAction.CWD:
				// Keep `/cwd <args>` in the input so the search field and the
				// token stay two-way bound while the picker is open.
				workingDirectoryQuery = args.trim();
				isWorkingDirectoryPickerOpen = true;
				break;
			case ChatFormCommandAction.MODEL:
				isWorkingDirectoryPickerOpen = false;
				opts.setValue('');
				opts.openModelSelector();
				break;
		}
	}

	function handleInput() {
		const value = opts.getValue();
		const cursor = opts.getCaretOffset() ?? value.length;

		if (value.startsWith(PROMPT_TRIGGER_PREFIX)) {
			isMentionPickerOpen = false;
			mentionQuery = '';
			isPromptPickerOpen = false;
			promptSearchQuery = '';

			const token = findCommandToken(value);
			if (!token) {
				isCommandPickerOpen = false;
				commandQuery = '';
				return;
			}

			// Dismissed token stays literal until it changes.
			const isDismissedSticky =
				commandDismissedSnapshot !== null &&
				commandDismissedSnapshot.name === token.name &&
				commandDismissedSnapshot.args === token.args;

			if (isDismissedSticky) {
				isCommandPickerOpen = false;
				commandQuery = '';
				return;
			}

			// Name complete once a space follows; exact match dispatches
			// instantly, non-match falls through. Disabled never dispatches.
			const nameComplete = token.args.length > 0 || value.endsWith(' ');
			if (nameComplete) {
				const command = availableCommands.find((c) => c.name === token.name);
				if (command && !command.disabled) {
					isCommandPickerOpen = false;
					commandQuery = '';
					dispatchCommand(command, token.args);
					return;
				}
			}

			// Name incomplete or unmatched: show the picker only when there
			// is something to pick.
			if (availableCommands.length > 0) {
				isCommandPickerOpen = true;
				commandQuery = token.name;
			} else {
				isCommandPickerOpen = false;
				commandQuery = '';
			}
			return;
		}

		isCommandPickerOpen = false;
		commandQuery = '';
		if (commandDismissedSnapshot !== null) {
			commandDismissedSnapshot = null;
		}
		// A non-command edit abandons `/cwd`.
		if (isWorkingDirectoryPickerOpen) {
			isWorkingDirectoryPickerOpen = false;
		}

		const token = findMentionToken(value, cursor);

		if (token) {
			// Dismissed token stays literal: don't reopen until it changes.
			const isDismissedSticky =
				mentionDismissedSnapshot !== null &&
				mentionDismissedSnapshot.start === token.start &&
				mentionDismissedSnapshot.query === token.query;

			if (!isDismissedSticky) {
				// Only search once a char follows `@`; a bare `@` is a no-op
				// (otherwise the picker flashes an empty hint on re-type).
				if (token.query.length > 0) {
					mentionDismissedSnapshot = null;
					isMentionPickerOpen = true;
					mentionQuery = token.query;
					isPromptPickerOpen = false;
					promptSearchQuery = '';
					return;
				}
			}
		}

		isPromptPickerOpen = false;
		promptSearchQuery = '';
		isMentionPickerOpen = false;
		mentionQuery = '';

		// Token gone or changed: reset the snapshot so a fresh `@` reopens.
		if (mentionDismissedSnapshot !== null && !token) {
			mentionDismissedSnapshot = null;
		}
	}

	function handleKeydown(event: KeyboardEvent): boolean {
		if (opts.getPickersRef()?.handleKeydown(event)) {
			return true;
		}

		if (event.key === KeyboardKey.ESCAPE && isPromptPickerOpen) {
			isPromptPickerOpen = false;
			promptSearchQuery = '';
			return true;
		}

		return false;
	}

	function handleCommandSelect(command: ChatFormCommand) {
		// Complete the command name with a trailing space and let the normal
		// input flow dispatch it (Enter on `/cw` behaves like typing `/cwd `).
		opts.setValue(`/${command.name} `);
		handleInput();
	}

	/**
	 * Command picker dismissed: snapshot the live `(name, args)` token so
	 * the same token stays literal until deleted or retyped.
	 */
	function handleCommandPickerClose() {
		if (isCommandPickerOpen) {
			commandDismissedSnapshot = takeCommandDismissSnapshot(opts.getValue());
		}
		isCommandPickerOpen = false;
		commandQuery = '';
		// Target picker manages its own focus: don't yank it back to the input.
		if (!isPromptPickerOpen && !isMentionPickerOpen && !isWorkingDirectoryPickerOpen) {
			opts.focusInput();
		}
	}

	/**
	 * Mention picker dismissed: snapshot the live `(start, query)` token
	 * so the same token stays literal until deleted or retyped.
	 */
	function handleMentionPickerClose() {
		if (isMentionPickerOpen) {
			const cursor = opts.getCaretOffset() ?? opts.getValue().length;
			mentionDismissedSnapshot = takeMentionDismissSnapshot(opts.getValue(), cursor);
		}
		isMentionPickerOpen = false;
		mentionQuery = '';
		opts.focusInput();
	}

	function handlePromptPickerClose() {
		isPromptPickerOpen = false;
		promptSearchQuery = '';
		opts.focusInput();
	}

	function handleWorkingDirectoryOpen() {
		workingDirectoryQuery = opts.getCwd() ?? '';
		isWorkingDirectoryPickerOpen = true;
	}

	function handleWorkingDirectoryClose() {
		isWorkingDirectoryPickerOpen = false;
		workingDirectoryQuery = '';
		opts.focusInput();
	}

	// Two-way bind the text after `/cwd ` and the picker search input; the
	// reverse direction is handled by handleInput -> dispatchCommand.
	$effect(() => {
		if (!isWorkingDirectoryPickerOpen) return;
		const value = opts.getValue();
		const token = findCommandToken(value);
		if (!token || token.name !== 'cwd') return;
		const newValue = `/cwd ${workingDirectoryQuery}`;
		if (newValue === value) return;
		opts.setValue(newValue);
		queueMicrotask(() => opts.setCaretOffset(newValue.length));
	});

	return {
		// Reactive picker state, exposed as getter/setter pairs so the chat
		// form can read them in the template and bind two-way where needed.
		get isCommandPickerOpen() {
			return isCommandPickerOpen;
		},
		set isCommandPickerOpen(v: boolean) {
			isCommandPickerOpen = v;
		},
		get commandQuery() {
			return commandQuery;
		},
		set commandQuery(v: string) {
			commandQuery = v;
		},
		get isPromptPickerOpen() {
			return isPromptPickerOpen;
		},
		set isPromptPickerOpen(v: boolean) {
			isPromptPickerOpen = v;
		},
		get promptSearchQuery() {
			return promptSearchQuery;
		},
		set promptSearchQuery(v: string) {
			promptSearchQuery = v;
		},
		get isMentionPickerOpen() {
			return isMentionPickerOpen;
		},
		set isMentionPickerOpen(v: boolean) {
			isMentionPickerOpen = v;
		},
		get mentionQuery() {
			return mentionQuery;
		},
		set mentionQuery(v: string) {
			mentionQuery = v;
		},
		get isWorkingDirectoryPickerOpen() {
			return isWorkingDirectoryPickerOpen;
		},
		set isWorkingDirectoryPickerOpen(v: boolean) {
			isWorkingDirectoryPickerOpen = v;
		},
		get workingDirectoryQuery() {
			return workingDirectoryQuery;
		},
		set workingDirectoryQuery(v: string) {
			workingDirectoryQuery = v;
		},
		get availableCommands() {
			return availableCommands;
		},
		get mentionScopePath() {
			return mentionScopePath;
		},
		handleInput,
		// Returns true when the event was consumed by a picker, so the chat
		// form can skip its own submit handling.
		handleKeydown,
		dispatchCommand,
		handleCommandSelect,
		handleCommandPickerClose,
		handleMentionPickerClose,
		handlePromptPickerClose,
		handleWorkingDirectoryOpen,
		handleWorkingDirectoryClose,
		openPromptPicker() {
			isPromptPickerOpen = true;
		},
		closePromptPicker() {
			isPromptPickerOpen = false;
			promptSearchQuery = '';
		}
	};
}

export type UseChatFormPickersReturn = ReturnType<typeof useChatFormPickers>;
