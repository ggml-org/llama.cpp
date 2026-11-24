import { loadServerConfig } from '#lib/utils/server-config.ts';

/** Runs once before the app starts: the server's address has to be known before the first request. */
export const init = () => loadServerConfig();
