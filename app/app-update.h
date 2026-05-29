#pragma once

// runs the self update flow against the configured release channel
// assume_yes: skip the interactive confirmation
int llama_app_update(bool assume_yes);

// dispatcher entry for the update command, parses -y/--yes
int llama_update(int argc, char ** argv);

// drops the previous binary left aside by a past windows update, no op elsewhere
void llama_app_startup(void);
