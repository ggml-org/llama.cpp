import { PROMPT_TRIGGER_PREFIX } from '$lib/constants';
import { ChatFormCommandAction, KeyboardKey } from '$lib/enums';
import type { ChatFormCommand } from '$lib/types';
import { getChatCommands } from '$lib/utils';
import {
	type CommandDismissSnapshot,
	findCommandToken,
	findMentionToken,
	type MentionDismissSnapshot,
	takeCommandDismissSnapshot,
	takeMentionDismissSnapshot
} from '$lib/utils';

	/** Injected getters avoid store circular imports. */
export interface UseChatFormPickersOptions {
	getValue: () => string;
	/** Also fires the form change handler. */
	setValue: (value: string) => void;
	/** Undefined after unmount. */
	getCaretOffset: () => number | undefined;
	setCaretOffset: (offset: number) => void;
	focusInput: () => void;
	/** Enables `/model`. */
	getShowModelSelector: () => boolean;
	/** Enables `/prompt`. */
	hasPrompts: () => boolean;
	/** Enables `/cwd`. */
	hasCwdTools: () => boolean;
	/** Enables `/skills`. */
	hasSkills: () => boolean;
	/** Dispatch a selected Skills command. */
	dispatchSkillsCommand: (args: string) => void;
	getCwd: () => string | null;
	/** Fallback scope for mention search. */
	getServerHome: () => string | null;
	openModelSelector: () => void;
	/** Delegate keydown to the mounted picker component. */
	getPickersRef: () => { handleKeydown(event: KeyboardEvent): boolean } | undefined;
}

/** Manage picker state and `/`/`@` routing. */
export function useChatFormPickers(opts: UseChatFormPickersOptions) {
	let isCommandPickerOpen = $state(false);
	let commandQuery = $state('');
	let isPromptPickerOpen = $state(false);
	let promptSearchQuery = $state('');
	let isMentionPickerOpen = $state(false);
	let mentionQuery = $state('');
	let isWorkingDirectoryPickerOpen = $state(false);
	let workingDirectoryQuery = $state('');
	let isSkillPickerOpen = $state(false);
	let skillQuery = $state('');
	// Keep dismissed mention tokens literal until edited.
	let mentionDismissedSnapshot: MentionDismissSnapshot | null = null;
	// Apply the same dismissal rule to slash commands.
	let commandDismissedSnapshot: CommandDismissSnapshot | null = null;

	// Use the server home until a CWD is selected.
	const mentionScopePath = $derived(opts.getCwd() ?? opts.getServerHome() ?? null);
	const availableCommands = $derived(
		getChatCommands({
			hasCwdTools: opts.hasCwdTools,
			hasPrompts: opts.hasPrompts,
			hasSkills: opts.hasSkills,
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
			case ChatFormCommandAction.PROMPT:
				isWorkingDirectoryPickerOpen = false;
				opts.setValue('');
				isPromptPickerOpen = true;
				promptSearchQuery = args.trim();

				break;
			case ChatFormCommandAction.CWD: {
				// Keep `/cwd <args>` in the input so the search field and the
				// token stay two-way bound; normalize partial tokens (`/cw foo`).
				const trimmed = args.trim();
				const newValue = `/cwd ${trimmed}`;

				if (opts.getValue() !== newValue) {
					opts.setValue(newValue);
					queueMicrotask(() => opts.setCaretOffset(newValue.length));
				}

				workingDirectoryQuery = trimmed;
				isWorkingDirectoryPickerOpen = true;

				break;
			}
			case ChatFormCommandAction.MODEL:
				isWorkingDirectoryPickerOpen = false;
				opts.setValue('');
				opts.openModelSelector();

				break;
			case ChatFormCommandAction.SKILLS: {
				const trimmed = args.trim();

				if (!trimmed) {
					opts.setValue('');
					opts.dispatchSkillsCommand('');

					break;
				}

				const newValue = `/skills ${trimmed}`;

				if (opts.getValue() !== newValue) {
					opts.setValue(newValue);
					queueMicrotask(() => opts.setCaretOffset(newValue.length));
				}

				skillQuery = trimmed;
				isSkillPickerOpen = true;

				break;
			}
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

			// The `/skills` picker triggers on a trailing space (`/skills `,
			// `/skills <name>`); a bare `/skills` stays on the command list
			// so Enter navigates to the catalog route.
			const hasTokenSpace =
				value.length > 1 + token.name.length && /\s/.test(value.charAt(1 + token.name.length));

			// While the `/cwd` picker is open the token doubles as its search
			// field: keep the two in sync instead of re-dispatching.
			if (isWorkingDirectoryPickerOpen) {
				isCommandPickerOpen = false;
				commandQuery = '';

				if (token.name === 'cwd') {
					workingDirectoryQuery = token.args.trim();
				} else {
					isWorkingDirectoryPickerOpen = false;
					workingDirectoryQuery = '';
				}

				return;
			}

			// Same controlled state for the `/skills` picker: the token is the
			// search field while open; editing it away closes the picker.
			// Never dispatches from this path. A bare `/skills` (trailing
			// space deleted) falls through to the command list so Enter still
			// navigates to the catalog.
			if (isSkillPickerOpen) {
				isCommandPickerOpen = false;
				commandQuery = '';

				if (token.name === 'skills' && hasTokenSpace) {
					skillQuery = token.args.trim();

					return;
				}

				isSkillPickerOpen = false;
				skillQuery = '';
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

			// `/skills` skips the command list: once the token name resolves
			// to `skills` and a space follows it, the typed name doubles as
			// the skill picker's fuzzy search field. Gated by the command's
			// availability; opens the picker only, never dispatches.
			if (
				token.name === 'skills' &&
				hasTokenSpace &&
				availableCommands.some((c) => c.name === 'skills' && !c.disabled)
			) {
				isCommandPickerOpen = false;
				commandQuery = '';
				isSkillPickerOpen = true;
				skillQuery = token.args.trim();

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

		if (isSkillPickerOpen) {
			isSkillPickerOpen = false;
			skillQuery = '';
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

	function handleWorkingDirectoryOpen() {
		workingDirectoryQuery = opts.getCwd() ?? '';
		isWorkingDirectoryPickerOpen = true;
	}

	function handleWorkingDirectoryClose() {
		isWorkingDirectoryPickerOpen = false;
		workingDirectoryQuery = '';
		opts.focusInput();
	}

	// Explicitly picked a skill: close the picker, clear the composer and
	// dispatch the durable activation exactly once. Runs only on explicit
	// selection, never from typing.
	function handleSkillSelect(name: string) {
		isSkillPickerOpen = false;
		skillQuery = '';
		// A successful activation ends the dismissed-token contract: the next
		// `/skills <query>` must rediscover instead of staying literal.
		commandDismissedSnapshot = null;
		opts.setValue('');
		opts.dispatchSkillsCommand(name);
	}

	// Picker dismissed (Escape/click-away): snapshot the live token so the
	// retained `/skills <query>` stays literal until deleted or retyped.
	function handleSkillPickerClose() {
		if (isSkillPickerOpen) {
			commandDismissedSnapshot = takeCommandDismissSnapshot(opts.getValue());
		}

		isSkillPickerOpen = false;
		skillQuery = '';
		opts.focusInput();
	}

	// Two-way bind the text after `/cwd ` and the picker search input; the
	// reverse direction is handled by handleInput.
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
		dispatchCommand,
		handleCommandPickerClose,
		handleCommandSelect,
		handleInput,
		// True when a picker consumed the event, so the form skips submit.
		handleKeydown,
		handleMentionPickerClose,
		handlePromptPickerClose,
		handleSkillPickerClose,
		handleSkillSelect,
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
		get isSkillPickerOpen() {
			return isSkillPickerOpen;
		},
		set isSkillPickerOpen(v: boolean) {
			isSkillPickerOpen = v;
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
		get skillQuery() {
			return skillQuery;
		},
		set skillQuery(v: string) {
			skillQuery = v;
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
