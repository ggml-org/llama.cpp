---
name: apple-reminders
description: Create, list, and complete Apple Reminders via the reminders CLI.
emoji: "⏰"
os: ["darwin"]
requires:
  bins: ["reminders"]
install:
  - brew install reminders
---

# Apple Reminders

Manage Apple Reminders from the command line using the `reminders` CLI.

## When to Use

- The user asks to add, list, complete, or delete a Reminder.
- A task should surface as a native Reminder synced across Apple devices.

## When NOT to Use

- The user wants a calendar event with a fixed time (use Calendar instead).
- Non-darwin platforms: the `reminders` CLI is macOS-only.

## Setup

Install the CLI once, then grant Reminders access when macOS prompts:

```
brew install reminders
reminders show "My List"
```

## Common Commands

```
reminders add "My List" "Buy oat milk"
reminders show "My List"
reminders complete "My List" 3
```
