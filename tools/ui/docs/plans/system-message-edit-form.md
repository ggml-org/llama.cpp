# Plan: Use `ChatMessageEditForm` for System Messages (library edge case)

## Goal

Extend the polished `ChatMessageEditForm.svelte` (currently used for user/assistant messages) so it also handles system messages. Today, system messages use a simpler inline edit UI inside `ChatMessageSystem.svelte` — basic textarea + "Add to Prompts library" checkbox + Cancel/Save. We want the same rich `ChatForm`-based editing experience for three "library" system-message cases:

1. Started from `/prompts` route (clicking "New chat" on a `PromptsCard` creates a conversation with a system message that already has a `CUSTOM_PROMPT` extra pointing at the library prompt).
2. Existing library-referenced system message inside a conversation (message has a `CUSTOM_PROMPT` extra whose referenced prompt still exists in `promptsStore`).
3. Fresh non-library system prompt used inside the conversation (plain system message that the user authored in-chat, no placeholder content, no `mcp:` instruction).

MCP-derived system messages continue to flow through `ChatMessageMcpPrompt` (which already uses `ChatMessageEditForm`). Placeholder content (`SYSTEM_MESSAGE_PLACEHOLDER`) keeps the existing inline UI so we can keep the "Add to Prompts library" CTA visually prominent on first edit.

## Current state (relevant pieces)

| File                                                                                                    | Role                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/lib/contexts/message-edit.context.ts`                                                              | Exposes edit state + actions to children (no system-specific fields today).                                                                                                                                |
| `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessage.svelte`                               | Owns edit state (`isEditing`, `editedContent`, `addToLibrary`, `deferSystemPromptSave`), renders `ChatMessageSystem` for `MessageRole.SYSTEM` (non-MCP). Save/cancel logic for system messages lives here. |
| `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageSystem/ChatMessageSystem.svelte`       | Renders the dashed display bubble + the simple inline edit UI + `ChatMessageActionIcons`.                                                                                                                  |
| `src/lib/components/app/chat/ChatMessages/ChatMessageEditForm.svelte`                                   | Renders the `ChatForm` wrapper + bottom row with role-specific toggles. Currently handles `USER` (save-only switch) and `ASSISTANT` (branch-after-edit switch).                                            |
| `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageMcpPrompt/ChatMessageMcpPrompt.svelte` | MCP-derived system/user messages: already uses `ChatMessageEditForm`.                                                                                                                                      |
| `src/lib/stores/chat.svelte.ts`                                                                         | `addSystemPrompt()`, `addSystemPromptWithContent()`, `removeSystemPromptPlaceholder()`, `pendingEditMessageId`.                                                                                            |
| `src/lib/stores/prompts.svelte.ts`                                                                      | `promptsStore.getPrompt(id)`, `addPrompt`, `updatePrompt`.                                                                                                                                                 |
| `src/lib/components/app/dialogs/DialogPromptAddNew.svelte`                                              | Title + content + category form for saving a system message as a new library prompt.                                                                                                                       |

## Changes

### 1. `message-edit.context.ts` — expose prompt-relationship fields

Add to `MessageEditState`:

```ts
readonly isSystemMessage: boolean;       // true when messageRole === SYSTEM
readonly isSystemPlaceholder: boolean;   // true when SYSTEM content === SYSTEM_MESSAGE_PLACEHOLDER
readonly isLibraryReferenced: boolean;   // true when message has CUSTOM_PROMPT extra
                                        // whose promptId is non-mcp AND the prompt still exists
readonly canAddToLibrary: boolean;       // true when user has not yet associated this with the library
                                        // i.e. !isLibraryReferenced && !placeholder
readonly libraryReferencedPrompt?: Prompt; // the Prompt object when isLibraryReferenced
```

Compute these in `ChatMessage.svelte`'s context `get` accessors using `customPromptExtra`, `referencedPrompt`, and `message.content === SYSTEM_MESSAGE_PLACEHOLDER`.

### 2. `ChatMessageEditForm.svelte` — support system mode

- Inside the `ChatForm`:
  - `showMcpPromptButton = false` for system messages (system prompts are bare text).
  - `showAddButton = false`, `showModelSelector = false` (no attachments, no per-message model override).
  - `placeholder = "Edit system message..."` for system.
- Bottom row (left side):
  - `ASSISTANT` -> existing "Branch conversation after edit" switch.
  - `USER` && `showSaveOnlyOption` -> existing "Update without re-sending" switch.
  - `SYSTEM` -> new branch:
    - When `canAddToLibrary`: "Add to Prompts library" checkbox (uses Svelte's `<Switch>`, same component as the others for visual consistency).
    - When `isLibraryReferenced`: muted label "Saved to library" + small link/button "Sync from library" if `promptIsStale` is true. (Sync could open `DialogPromptSync` via the existing context wiring.)
  - Else: empty `<div>` (mirror existing fallback).
- Move the "Add to library" toggle's state into the form (`let addToLibrary = $state(false)`) since `ChatMessage.svelte` will no longer own it. Keep a `bindable` via context or via a callback if `ChatMessage.svelte` still needs to react (it does — for `handleSaveEdit` to open `DialogPromptAddNew`).
- Right side: existing Cancel button.
- Save behavior:
  - `USER` + `saveWithoutRegenerate` -> `editCtx.saveOnly()`.
  - else -> `editCtx.save()`.
  - The save flow in `ChatMessage.svelte` already inspects `addToLibrary` via the callback (`onAddToLibraryChange`). We prop-forward that signal back up.

### 3. `ChatMessageSystem.svelte` — route edits through the form

- Remove the inline edit UI entirely (textarea + checkbox + Cancel/Save buttons).
- When `editCtx.isEditing`:
  - If `editCtx.isSystemPlaceholder` -> still try to use `<ChatMessageEditForm />` for consistency (placeholder uses the same form; the "Add to library" toggle will be visible because `canAddToLibrary` is true).
  - Else -> `<ChatMessageEditForm />`.
- Keep all existing display logic (dashed bubble, expand/collapse, promptId title row, "Newer version available", action icons) — only the edit block is replaced.
- Drop the now-unused props (`addToLibrary`, `onAddToLibraryChange`, `deferSystemPromptSave`, `textareaElement` if no longer used). Re-add `textareaElement` only if needed for focus management that the form's `ChatFormTextarea` doesn't already expose (it does — via `focus()`).

### 4. `ChatMessage.svelte` — wire state into the form

- Drop local `addToLibrary` / `deferSystemPromptSave` state (move to form, keep save-side handlers).
- In `handleSaveEdit` for `SYSTEM`:
  - Keep the existing "remove placeholder if empty" branch.
  - Keep the "open `DialogPromptAddNew` if `addToLibrary` is true" branch (now sourced from form's callback).
  - Otherwise, update message in place preserving non-CUSTOM_PROMPT extras.
- `DialogPromptAddNew.onAddToLibraryComplete` callback stays the same (saves system message with new `CUSTOM_PROMPT` extra pointing at the new prompt).
- Add a `systemMessagePickAddToLibrary` callback on the form (via context or prop) so the form can ask the parent to open `DialogPromptAddNew`.

### 5. Optional polish — visual coherence

- The system message display uses a dashed border; the new `ChatForm` does not. Two options:
  - (a) Wrap `<ChatMessageEditForm />` in the same dashed `Card` for symmetry.
  - (b) Accept the visual diff for the duration of the edit (consistent with how user/assistant edits already differ from display).
  - **Recommendation: (a)** — wrap with a dashed bubble matching the read view. We can pass a `variant="system"` prop to `ChatMessageEditForm` that toggles wrapping in a `Card` with the dashed style. Minimal CSS change, keeps the read/edit relationship obvious.

## Files to touch

1. `src/lib/contexts/message-edit.context.ts` — add system-prompt fields.
2. `src/lib/components/app/chat/ChatMessages/ChatMessageEditForm.svelte` — system branch in bottom row, tweaked `ChatForm` props, optional `variant` prop.
3. `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageSystem/ChatMessageSystem.svelte` — drop inline edit UI, render `<ChatMessageEditForm />`.
4. `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessage.svelte` — move library toggle into form, simplify save branch.

## Open questions for you

1. **Eligibility on first edit.** A new system message added in-chat starts with `SYSTEM_MESSAGE_PLACEHOLDER`. Two options:
   - (a) Placeholder uses the new `ChatMessageEditForm` too (with "Add to library" prominent). I prefer this for consistency.
   - (b) Placeholder keeps the inline UI; new form appears after the first save. Less consistent.
     Which do you want?

2. **"Sync from library" affordance.** Currently `ChatMessageSystem` shows "Newer version available" below the bubble as a link-triggered `DialogPromptSync`. Should the new edit form also surface this (e.g., when `promptIsStale` is true, add a "Sync" pill next to the bottom-row switch), or keep it at the bubble level?

3. **Visual variant.** Should `<ChatMessageEditForm />` render inside a dashed-bordered wrapper for `SYSTEM` to match the read view? (Recommended above.)

4. **Orphaned references.** If the user deletes a library prompt whose `CUSTOM_PROMPT` is still on a system message, `promptsStore.getPrompt(id)` returns `undefined`. The plan above treats that as `!isLibraryReferenced` (i.e., eligible for the new form, with `canAddToLibrary=true`). Confirm that is acceptable or specify what to surface instead.

5. **Scope.** Anything else missing from the three cases in your original brief?
