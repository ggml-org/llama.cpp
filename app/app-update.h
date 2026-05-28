#pragma once

// runs the self update flow against the configured release channel
// force: reinstall the channel build even when the local build is already current
int llama_app_update(bool force);

// dispatcher entry for the update command, parses an optional --force
int llama_update(int argc, char ** argv);

// best effort line printed at startup when the channel advertises a newer build
void llama_app_update_notice(void);

// drops the previous binary left aside by a past windows update, no op elsewhere
void llama_app_startup(void);
