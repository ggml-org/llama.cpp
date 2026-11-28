import { CWD_COMMAND_NAME } from '#lib/constants/index.ts';
import { ChatFormCommandAction, KeyboardKey } from '#lib/enums/index.ts';
import type { ChatFormCommand } from '#lib/types/index.ts';
import {
	abbreviateHome,
	type CommandDismissSnapshot,
	type CommandToken,
	findCommandToken,
	findMentionToken,
	getChatCommands,
	insertCwdCommand,
	isCommandDismissed,
	locateCwdCommand,
	type MentionDismissSnapshot,
	removeCommandToken,
	replaceWithCwdCommand,
	takeCommandDismissSnapshot,
	takeMentionDismissSnapshot
} from '#lib/utils/index.ts';

/** Dependencies injected as getters so the hook stays free of store circular imports. */
export interface UseChatFormPickersOptions {
	getValue: () => string;
	/** Also fires the form's onChange. */
	setValue: (value: string) => void;
	/** Undefined when unmounted. */
	getCaretOffset: () => number | undefined;
	setCaretOffset: (offset: number) => void;
	focusInput: () => void;
	/** Whether typing `/` or `@` opens a picker at all. */
	opensOnInput: () => boolean;
	/**
	 * Whether the working directory is typed in the chat input, after a `/cwd` command
	 * wherever it is in the message, instead of in the picker's own search field.
	 */
	editsCwdInline: () => boolean;
	/** Gates `/model`. */
	getShowModelSelector: () => boolean;
	/** Gates `/prompt`. */
	hasPrompts: () => boolean;
	/** Gates `/cwd`. */
	hasCwdTools: () => boolean;
	/** Gates `/compact`. */
	canCompact: () => boolean;
	/** Runs `/compact`. */
	onCompact: () => void;
	/** Gates `/system-message`. */
	canEditSystemMessage: () => boolean;
	/** Runs `/system-message`: the form turns into the system message editor. */
	onSystemMessage: () => void;
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
	// The open picker searches the `/cwd` command in the chat input, not a field of its own.
	// Set on every open and kept after a close, so the picker fades out the way it looked
	let isWorkingDirectoryInline = $state(false);
	// Last dismissed `@`-mention token; while intact, the picker does not
	// reopen, so an escaped `@<query>` stays literal until edited.
	let mentionDismissedSnapshot: MentionDismissSnapshot | null = null;
	// Same dismissal contract for the `/`-command token.
	let commandDismissedSnapshot: CommandDismissSnapshot | null = null;
	// Where the open working directory picker's `/cwd` command is, its path as last typed,
	// and the input's length then: an edit elsewhere in the message moves it by the
	// change in length if it was made before it
	let cwdCommand: { start: number; query: string; length: number } | null = null;

	// Fall back to the server home so the picker still finds matches
	// before a cwd is set.
	const mentionScopePath = $derived(opts.getCwd() ?? opts.getServerHome() ?? null);
	const availableCommands = $derived(
		getChatCommands({
			canCompact: opts.canCompact,
			canEditSystemMessage: opts.canEditSystemMessage,
			hasCwdTools: opts.hasCwdTools,
			hasPrompts: opts.hasPrompts,
			showModelSelector: opts.getShowModelSelector()
		})
	);
	// only the commands that take an argument read the words after them
	const commandNames = $derived(
		availableCommands.filter((command) => command.takesArgument).map((command) => command.name)
	);
	const caretOf = (value: string) => opts.getCaretOffset() ?? value.length;
	const commandAt = (value: string) => findCommandToken(value, caretOf(value), commandNames);

	/** The open picker's `/cwd` command where it now is, or null when it was edited away. */
	function currentCwdCommand(value = opts.getValue()) {
		if (!cwdCommand) return null;

		const { length, query, start } = cwdCommand;
		const shift = value.length - length;
		const caret = opts.getCaretOffset();
		// an edit made before the command leaves the caret before it too
		const isEditBefore = caret === undefined || caret <= start + shift;

		return locateCwdCommand(value, start, query, isEditBefore ? shift : 0);
	}

	function trackCwdCommand(start: number, query: string, value: string) {
		cwdCommand = { length: value.length, query, start };
	}

	function setValueAndCaret(value: string, caret: number) {
		if (opts.getValue() !== value) opts.setValue(value);

		queueMicrotask(() => opts.setCaretOffset(caret));
	}

	/** Takes the command out of the message, leaving the words around it. */
	function consumeCommand(token: { start: number; end: number }) {
		const next = removeCommandToken(opts.getValue(), token);

		setValueAndCaret(next.value, next.caret);
	}

	// Dispatch a slash command picked from the list, wherever it is in the message: take the
	// token out and open the target picker, seeding its search with `args`. Runs only on
	// explicit selection (Enter/click), so nothing is cleared mid-typing.
	function dispatchCommand(command: ChatFormCommand, token: CommandToken) {
		isCommandPickerOpen = false;
		commandQuery = '';

		if (command.action !== ChatFormCommandAction.CWD) {
			isWorkingDirectoryPickerOpen = false;
			cwdCommand = null;
			consumeCommand(token);
		}

		switch (command.action) {
			case ChatFormCommandAction.COMPACT:
				opts.onCompact();

				break;
			case ChatFormCommandAction.PROMPT:
				isPromptPickerOpen = true;
				promptSearchQuery = token.args.trim();

				break;
			case ChatFormCommandAction.CWD: {
				// `/cwd <path>` stays where it was typed, the path two-way bound to the picker's
				// search; a partial token (`/cw foo`) is completed
				const query = token.args.trim();
				const next = replaceWithCwdCommand(opts.getValue(), token, query);

				isWorkingDirectoryInline = opts.editsCwdInline();
				trackCwdCommand(next.start, query, next.value);
				setValueAndCaret(next.value, next.caret);
				workingDirectoryQuery = query;
				isWorkingDirectoryPickerOpen = true;

				break;
			}
			case ChatFormCommandAction.MODEL:
				opts.openModelSelector();

				break;
			case ChatFormCommandAction.SYSTEM_MESSAGE:
				opts.onSystemMessage();

				break;
		}
	}

	function handleInput() {
		if (!opts.opensOnInput()) return;

		const value = opts.getValue();
		const cursor = caretOf(value);
		const command = commandAt(value);

		// While the `/cwd` picker is open the command doubles as its search field: keep the two
		// in sync instead of re-dispatching. An edit elsewhere in the message leaves it open
		if (isWorkingDirectoryPickerOpen && cwdCommand) {
			isCommandPickerOpen = false;
			commandQuery = '';

			// the path as typed when the caret is in the command, not in another `/cwd` the
			// message names
			const tracked = currentCwdCommand(value);
			const located =
				tracked && command?.name === CWD_COMMAND_NAME && command.start === tracked.start
					? { query: command.args, start: command.start }
					: tracked;

			if (located) {
				trackCwdCommand(located.start, located.query, value);
				workingDirectoryQuery = located.query.trim();
			} else {
				isWorkingDirectoryPickerOpen = false;
				workingDirectoryQuery = '';
				cwdCommand = null;
			}

			return;
		}

		if (command) {
			isMentionPickerOpen = false;
			mentionQuery = '';
			isPromptPickerOpen = false;
			promptSearchQuery = '';

			// Dismissed token stays literal until it changes.
			const isDismissedSticky = isCommandDismissed(commandDismissedSnapshot, command);

			if (isDismissedSticky) {
				isCommandPickerOpen = false;
				commandQuery = '';

				return;
			}

			// Commands dispatch only on explicit selection (Enter/click),
			// never mid-typing: `/model is broken` is prose until the user
			// picks the command from the list.
			if (availableCommands.length > 0) {
				isCommandPickerOpen = true;
				commandQuery = command.name;
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
		// Dispatch on the live token so typed args seed the target picker.
		const value = opts.getValue();
		const caret = caretOf(value);
		const token = commandAt(value) ?? { args: '', end: caret, name: '', start: caret };

		dispatchCommand(command, token);
	}

	// Picker dismissed: snapshot the live token so it stays literal until
	// deleted or retyped.
	function handleCommandPickerClose() {
		if (isCommandPickerOpen) {
			const value = opts.getValue();

			commandDismissedSnapshot = takeCommandDismissSnapshot(value, caretOf(value), commandNames);
		}

		isCommandPickerOpen = false;
		commandQuery = '';

		// Target picker manages its own focus: don't yank it back to the input.
		if (!isPromptPickerOpen && !isMentionPickerOpen && !isWorkingDirectoryPickerOpen) {
			opts.focusInput();
		}
	}

	// Same dismissal snapshot for the mention token.
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

	/** True while the caret is in the `/cwd` command, not in the message around it. */
	function isCaretInCwdCommand(): boolean {
		const located = currentCwdCommand();
		const caret = opts.getCaretOffset();

		return !located || caret === undefined || (caret >= located.start && caret <= located.end);
	}

	/** Takes the `/cwd` command out, leaving the message written around it. */
	function consumeCwdCommand() {
		const located = currentCwdCommand();

		cwdCommand = null;

		if (located) consumeCommand(located);
	}

	// Opened from the + menu or the chip. Inline, the current directory (or the path already
	// after a `/cwd` command) goes on a `/cwd` command at the caret, to be edited there
	function handleWorkingDirectoryOpen() {
		const cwd = opts.getCwd();
		const seed = cwd ? abbreviateHome(cwd, opts.getServerHome()) : '';

		isWorkingDirectoryInline = opts.editsCwdInline();

		if (isWorkingDirectoryInline) {
			const value = opts.getValue();
			const located = currentCwdCommand(value);

			if (located) {
				trackCwdCommand(located.start, located.query, value);
				workingDirectoryQuery = located.query.trim();
				queueMicrotask(() => opts.setCaretOffset(located.end));
			} else {
				const next = insertCwdCommand(value, caretOf(value), seed);

				trackCwdCommand(next.start, seed, next.value);
				workingDirectoryQuery = seed;
				setValueAndCaret(next.value, next.caret);
			}
		} else {
			workingDirectoryQuery = seed;
		}

		isWorkingDirectoryPickerOpen = true;
	}

	// A closed picker leaves no `/cwd` command behind: committing a directory or dismissing
	// the picker gives the message back as it was
	function handleWorkingDirectoryClose() {
		isWorkingDirectoryPickerOpen = false;
		workingDirectoryQuery = '';
		consumeCwdCommand();
		opts.focusInput();
	}

	// Two-way bind the text after `/cwd ` and the picker's own search field; the reverse
	// direction is handled by handleInput. Inline, the `/cwd` command is the only field
	$effect(() => {
		const query = workingDirectoryQuery;

		if (!isWorkingDirectoryPickerOpen || isWorkingDirectoryInline) return;

		const located = currentCwdCommand();

		if (!located || located.query === query) return;

		const next = replaceWithCwdCommand(opts.getValue(), located, query);

		trackCwdCommand(next.start, query, next.value);
		opts.setValue(next.value);
	});

	return {
		get availableCommands() {
			return availableCommands;
		},
		closePromptPicker() {
			isPromptPickerOpen = false;
			promptSearchQuery = '';
		},
		get commandQuery() {
			return commandQuery;
		},
		set commandQuery(v: string) {
			commandQuery = v;
		},
		consumeCwdCommand,
		dispatchCommand,
		handleCommandPickerClose,
		handleCommandSelect,
		handleInput,
		// True when a picker consumed the event, so the form skips submit.
		handleKeydown,
		handleMentionPickerClose,
		handlePromptPickerClose,
		handleWorkingDirectoryClose,
		handleWorkingDirectoryOpen,
		isCaretInCwdCommand,
		get isCommandPickerOpen() {
			return isCommandPickerOpen;
		},
		set isCommandPickerOpen(v: boolean) {
			isCommandPickerOpen = v;
		},
		get isMentionPickerOpen() {
			return isMentionPickerOpen;
		},
		set isMentionPickerOpen(v: boolean) {
			isMentionPickerOpen = v;
		},
		get isPromptPickerOpen() {
			return isPromptPickerOpen;
		},
		set isPromptPickerOpen(v: boolean) {
			isPromptPickerOpen = v;
		},
		get isWorkingDirectoryInline() {
			return isWorkingDirectoryInline;
		},
		get isWorkingDirectoryPickerOpen() {
			return isWorkingDirectoryPickerOpen;
		},
		set isWorkingDirectoryPickerOpen(v: boolean) {
			isWorkingDirectoryPickerOpen = v;
		},
		get mentionQuery() {
			return mentionQuery;
		},
		set mentionQuery(v: string) {
			mentionQuery = v;
		},
		get mentionScopePath() {
			return mentionScopePath;
		},
		openPromptPicker() {
			isPromptPickerOpen = true;
		},
		get promptSearchQuery() {
			return promptSearchQuery;
		},
		set promptSearchQuery(v: string) {
			promptSearchQuery = v;
		},
		get workingDirectoryQuery() {
			return workingDirectoryQuery;
		},
		set workingDirectoryQuery(v: string) {
			workingDirectoryQuery = v;
		}
	};
}

export type UseChatFormPickersReturn = ReturnType<typeof useChatFormPickers>;
