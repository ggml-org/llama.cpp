import { CWD_COMMAND_LINE_BREAK, PROMPT_TRIGGER_PREFIX } from '#lib/constants/index.ts';
import { ChatFormCommandAction, KeyboardKey } from '#lib/enums/index.ts';
import type { ChatFormCommand } from '#lib/types/index.ts';
import {
	abbreviateHome,
	type CommandDismissSnapshot,
	composeCwdCommandLine,
	cwdCommandCaretOffset,
	findCommandToken,
	findMentionToken,
	getChatCommands,
	type MentionDismissSnapshot,
	parseCwdCommandLine,
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
	 * Whether the working directory is typed in the chat input, on a `/cwd` line above the
	 * message, instead of in the picker's own search field.
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
	// The open picker searches the `/cwd` line of the chat input, not a field of its own. Set
	// on every open and kept after a close, so the picker fades out the way it looked
	let isWorkingDirectoryInline = $state(false);
	// Last dismissed `@`-mention token; while intact, the picker does not
	// reopen, so an escaped `@<query>` stays literal until edited.
	let mentionDismissedSnapshot: MentionDismissSnapshot | null = null;
	// Same dismissal contract for the `/`-command token.
	let commandDismissedSnapshot: CommandDismissSnapshot | null = null;

	// Fall back to the server home so the picker still finds matches
	// before a cwd is set.
	const mentionScopePath = $derived(opts.getCwd() ?? opts.getServerHome() ?? null);
	const availableCommands = $derived(
		getChatCommands({
			canCompact: opts.canCompact,
			hasCwdTools: opts.hasCwdTools,
			hasPrompts: opts.hasPrompts,
			showModelSelector: opts.getShowModelSelector()
		})
	);

	// Dispatch a slash command picked from the list: consume the token and
	// open the target picker, seeding its search with `args`. Runs only on
	// explicit selection (Enter/click), so the buffer is never cleared
	// mid-typing.
	function dispatchCommand(command: ChatFormCommand, args: string) {
		isCommandPickerOpen = false;
		commandQuery = '';

		switch (command.action) {
			case ChatFormCommandAction.COMPACT:
				isWorkingDirectoryPickerOpen = false;
				opts.setValue('');
				opts.onCompact();

				break;
			case ChatFormCommandAction.PROMPT:
				isWorkingDirectoryPickerOpen = false;
				opts.setValue('');
				isPromptPickerOpen = true;
				promptSearchQuery = args.trim();

				break;
			case ChatFormCommandAction.CWD: {
				// Keep `/cwd <path>` on the input's first line, the message below it, so the
				// path and the picker's search stay two-way bound; normalize partial tokens
				// (`/cw foo`)
				const lineEnd = args.indexOf(CWD_COMMAND_LINE_BREAK);
				const query = (lineEnd === -1 ? args : args.slice(0, lineEnd)).trim();
				const message = lineEnd === -1 ? '' : args.slice(lineEnd + 1);

				isWorkingDirectoryInline = opts.editsCwdInline();
				showCwdLine(query, message);
				workingDirectoryQuery = query;
				isWorkingDirectoryPickerOpen = true;

				break;
			}
			case ChatFormCommandAction.MODEL:
				isWorkingDirectoryPickerOpen = false;
				opts.setValue('');
				opts.openModelSelector();

				break;
		}
	}

	function handleInput() {
		if (!opts.opensOnInput()) return;

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

			// While the `/cwd` picker is open the token doubles as its search
			// field: keep the two in sync instead of re-dispatching.
			if (isWorkingDirectoryPickerOpen) {
				isCommandPickerOpen = false;
				commandQuery = '';

				const cwdLine = parseCwdCommandLine(value);

				if (cwdLine) {
					workingDirectoryQuery = cwdLine.query.trim();
				} else {
					isWorkingDirectoryPickerOpen = false;
					workingDirectoryQuery = '';
				}

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

			// Commands dispatch only on explicit selection (Enter/click),
			// never mid-typing: `/model is broken` is prose until the user
			// picks the command from the list.
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
		// Dispatch on the live token so typed args seed the target picker.
		const token = findCommandToken(opts.getValue());

		dispatchCommand(command, token?.args ?? '');
	}

	// Picker dismissed: snapshot the live token so it stays literal until
	// deleted or retyped.
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

	/**
	 * Puts `/cwd <query>` above the message. Inline, the caret goes to the end of the path;
	 * otherwise the picker's own field keeps the focus.
	 */
	function showCwdLine(query: string, message: string) {
		const next = composeCwdCommandLine(query, message);

		if (opts.getValue() !== next) opts.setValue(next);

		if (isWorkingDirectoryInline) {
			queueMicrotask(() => opts.setCaretOffset(cwdCommandCaretOffset(query)));
		}
	}

	/** Drops the `/cwd` line, leaving the message written below it. */
	function consumeCwdCommand() {
		const cwdLine = parseCwdCommandLine(opts.getValue());

		if (!cwdLine) return;

		opts.setValue(cwdLine.message);
		queueMicrotask(() => opts.setCaretOffset(cwdLine.message.length));
	}

	// Opened from the + menu or the chip. Inline, the current directory (or the path already
	// on a `/cwd` line) goes on a `/cwd` line above the message, to be edited there
	function handleWorkingDirectoryOpen() {
		const cwd = opts.getCwd();
		const seed = cwd ? abbreviateHome(cwd, opts.getServerHome()) : '';

		isWorkingDirectoryInline = opts.editsCwdInline();

		if (isWorkingDirectoryInline) {
			const value = opts.getValue();
			const cwdLine = parseCwdCommandLine(value);
			const query = cwdLine ? cwdLine.query.trim() : seed;

			showCwdLine(query, cwdLine ? cwdLine.message : value);
			workingDirectoryQuery = query;
		} else {
			workingDirectoryQuery = seed;
		}

		isWorkingDirectoryPickerOpen = true;
	}

	// A closed picker leaves no `/cwd` line behind: committing a directory or dismissing the
	// picker gives the message back as it was
	function handleWorkingDirectoryClose() {
		isWorkingDirectoryPickerOpen = false;
		workingDirectoryQuery = '';
		consumeCwdCommand();
		opts.focusInput();
	}

	// Two-way bind the text after `/cwd ` and the picker's own search field; the reverse
	// direction is handled by handleInput. Inline, the `/cwd` line is the only field
	$effect(() => {
		if (!isWorkingDirectoryPickerOpen || isWorkingDirectoryInline) return;

		const cwdLine = parseCwdCommandLine(opts.getValue());

		if (!cwdLine || cwdLine.query === workingDirectoryQuery) return;

		showCwdLine(workingDirectoryQuery, cwdLine.message);
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
