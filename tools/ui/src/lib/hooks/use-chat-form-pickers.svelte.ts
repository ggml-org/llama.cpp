import { getChatCommands, PROMPT_TRIGGER_PREFIX } from '$lib/constants';
import { KeyboardKey } from '$lib/enums';
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
 * Cross-cutting dependencies the pickers need from the chat form. Injected
 * as getters/callbacks so the hook stays free of component-owned state
 * (the input value, caret, model selector) and store circular imports.
 */
export interface UseChatFormPickersOptions {
	/** Current chat input value. */
	getValue: () => string;
	/** Assign a new chat input value (also fires the form's onChange). */
	setValue: (value: string) => void;
	/** Live caret offset in the active input; undefined when unmounted. */
	getCaretOffset: () => number | undefined;
	/** Move the caret in the active input. */
	setCaretOffset: (offset: number) => void;
	/** Refocus the chat input after a picker closes. */
	focusInput: () => void;
	/** Whether the model selector is rendered (gates `/model`). */
	getShowModelSelector: () => boolean;
	/** Predicate: MCP prompts are reachable (gates `/prompt`). */
	hasPrompts: () => boolean;
	/** Predicate: built-in tools present (gates `/cwd`). */
	hasBuiltinTools: () => boolean;
	/** Predicate: at least one recent mention exists (recents surface). */
	hasRecents: () => boolean;
	/** Current working directory, if the user picked one. */
	getCwd: () => string | null;
	/** Server home directory (mention search fallback scope). */
	getServerHome: () => string | null;
	/** Open the model selector (dispatched by `/model`). */
	openModelSelector: () => void;
	/** Delegate a keydown to the mounted pickers component, if any. */
	getPickersRef: () => { handleKeydown(event: KeyboardEvent): boolean } | undefined;
}

/**
 * State and orchestration for the chat form's pickers and the `/`+`@`
 * routing that drives them.
 *
 * Owns the open/query state for the command, prompt, mention and working-
 * directory pickers, the dismiss snapshots, the slash-command dispatch, and
 * the two-way `/cwd` token binding. The plain textarea/contenteditable swap,
 * caret pinning and file/attachment handling stay in the chat form.
 */
export function useChatFormPickers(opts: UseChatFormPickersOptions) {
	// Picker state
	let isCommandPickerOpen = $state(false);
	let commandQuery = $state('');
	let isPromptPickerOpen = $state(false);
	let promptSearchQuery = $state('');
	let isMentionPickerOpen = $state(false);
	let mentionQuery = $state('');
	let isWorkingDirectoryPickerOpen = $state(false);
	let workingDirectoryQuery = $state('');

	/**
	 * Snapshot of the most recent `@`-mention token the user dismissed
	 * (via Escape, outside-click, or simply by deleting it). When the
	 * picker is closed AND the same token is still intact in the buffer,
	 * we do NOT auto-reopen - the user has explicitly told us this
	 * `@<query>` should be treated as literal text. The snapshot
	 * becomes stale the moment any character inside the token changes,
	 * at which point the picker is allowed to reopen on the next input.
	 */
	let mentionDismissedSnapshot: MentionDismissSnapshot | null = null;

	/**
	 * Snapshot of the most recent `/`-command token the user dismissed
	 * (via Escape or outside-click). When the command picker is closed AND
	 * the same token is still intact in the buffer, we neither reopen the
	 * picker nor instant-dispatch - the user has explicitly told us this
	 * `/name` should be treated as literal text. The snapshot becomes stale
	 * the moment any character inside the token changes.
	 */
	let commandDismissedSnapshot: CommandDismissSnapshot | null = null;

	// Scopes the @-mention search to the cwd the user picked (when set),
	// falling back to the server home so the picker still finds matches
	// before the user has chosen a directory.
	const mentionScopePath = $derived(opts.getCwd() ?? opts.getServerHome() ?? null);

	// Slash commands surfaced by the `/` command picker, filtered to those
	// whose backing capability is currently available.
	const availableCommands = $derived(
		getChatCommands({
			showModelSelector: opts.getShowModelSelector(),
			hasPrompts: opts.hasPrompts,
			hasBuiltinTools: opts.hasBuiltinTools
		})
	);

	/**
	 * Dispatch a selected slash command. The command token is consumed
	 * (the input is cleared) and the corresponding picker / selector is
	 * opened. `args` (everything after the command name) seeds the target
	 * picker's search where applicable - e.g. `/prompt rev` opens the MCP
	 * prompt picker pre-filtered by `rev`.
	 */
	function dispatchCommand(command: ChatFormCommand, args: string) {
		switch (command.action) {
			case 'prompt':
				isWorkingDirectoryPickerOpen = false;
				opts.setValue('');
				isPromptPickerOpen = true;
				promptSearchQuery = args.trim();
				break;
			case 'cwd':
				// Keep `/cwd <args>` in the input so the search field and the
				// token stay two-way bound while the picker is open.
				workingDirectoryQuery = args.trim();
				isWorkingDirectoryPickerOpen = true;
				break;
			case 'model':
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
			// A `/` at the start is a command, not a mention - close the
			// mention and prompt pickers and route to the command picker.
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

			// Picker's been dismissed for THIS exact token - honor the
			// "literal until delete + retype" rule: don't reopen or
			// instant-dispatch until the token changes.
			const isDismissedSticky =
				commandDismissedSnapshot !== null &&
				commandDismissedSnapshot.name === token.name &&
				commandDismissedSnapshot.args === token.args;

			if (isDismissedSticky) {
				isCommandPickerOpen = false;
				commandQuery = '';
				return;
			}

			// The command name is complete once a space follows it. An exact
			// match dispatches instantly (Slack-style); a non-match falls
			// through to the picker's empty state so the user can still
			// submit the literal text. Disabled commands never dispatch.
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

			// Still typing the name (or it doesn't match) - show the picker
			// only when there is something to pick.
			if (availableCommands.length > 0) {
				isCommandPickerOpen = true;
				commandQuery = token.name;
			} else {
				isCommandPickerOpen = false;
				commandQuery = '';
			}
			return;
		}

		// Not a command - close the command picker and reset the snapshot.
		isCommandPickerOpen = false;
		commandQuery = '';
		if (commandDismissedSnapshot !== null) {
			commandDismissedSnapshot = null;
		}
		// A non-command edit while the `/cwd` picker is open means the user
		// abandoned the command - close the picker.
		if (isWorkingDirectoryPickerOpen) {
			isWorkingDirectoryPickerOpen = false;
		}

		const token = findMentionToken(value, cursor);

		if (token) {
			// Picker's been dismissed for THIS exact token - honor the
			// "literal until delete + retype" rule: don't reopen until the
			// token changes (typed-then-Esc'd a slot, then kept typing
			// inside the same `@<q>`).
			const isDismissedSticky =
				mentionDismissedSnapshot !== null &&
				mentionDismissedSnapshot.start === token.start &&
				mentionDismissedSnapshot.query === token.query;

			if (!isDismissedSticky) {
				// Show the picker only if it can actually render something
				// useful: either the user has typed at least one
				// character after `@` (live search), or we've previously
				// picked at least one file/folder (recents surface). A
				// bare `@` with no recents is a no-op - re-typing into
				// the token would otherwise flash an empty "start
				// typing..." hint before the user types anything.
				const haveRecents = opts.hasRecents();
				const haveQuery = token.query.length > 0;

				if (haveRecents || haveQuery) {
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

		// Token gone or no longer intact - the snapshot is stale. Reset so
		// the next fresh `@` opens immediately even at the same offset.
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
		// Complete the command name in the input (with a trailing space) and
		// let the normal input flow dispatch it. This way `/cw` + Enter yields
		// `/cwd ` in the chat form, and the instant-dispatch-on-space path
		// opens the target picker exactly as if the user had typed it.
		opts.setValue(`/${command.name} `);
		handleInput();
	}

	/**
	 * Command picker dismissed (Esc, outside-click, or selection-complete).
	 * Capture a `(name, args)` snapshot of the live token so subsequent
	 * input events that produce the SAME token won't reopen the picker or
	 * instant-dispatch - the user has explicitly told us that `/name`
	 * should be literal until they delete or retype a fresh `/`.
	 */
	function handleCommandPickerClose() {
		if (isCommandPickerOpen) {
			commandDismissedSnapshot = takeCommandDismissSnapshot(opts.getValue());
		}
		isCommandPickerOpen = false;
		commandQuery = '';
		// When a command was selected, the target picker/selector takes over
		// and manages its own focus - don't yank focus back to the chat input.
		if (!isPromptPickerOpen && !isMentionPickerOpen && !isWorkingDirectoryPickerOpen) {
			opts.focusInput();
		}
	}

	/**
	 * Mention picker dismissed (Esc, outside-click, or selection-complete).
	 * Capture a `(start, query)` snapshot of the live token so subsequent
	 * input events that produce the SAME token won't reopen the picker -
	 * the user has explicitly told us that `@<query>` should be literal
	 * until they delete or retype a fresh `@`.
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

	// Two-way binding between the text after `/cwd ` and the picker's search
	// input: typing in the search input rewrites the `/cwd <query>` token in
	// the chat input. The reverse direction (typing in the chat input) is
	// handled by `handleInput` -> `dispatchCommand` -> `workingDirectoryQuery`.
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
