// First bytes of every ZIP local file header ("PK"). Import detects an archive
// from these bytes rather than from the filename, which the OS may not preserve.
export const ZIP_MAGIC = [0x50, 0x4b];

// The settings file inside a full backup ZIP, next to one JSONL file per conversation.
export const SETTINGS_ARCHIVE_ENTRY = 'settings.json';

// Filename of a full backup ZIP (conversations and settings), after its date.
export const BACKUP_ARCHIVE_NAME = 'llama_chat_backup';
