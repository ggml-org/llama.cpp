/**
 * Server role enum - used for single/multi-model mode
 */
export enum ServerRole {
	/** Single model mode - server running with a specific model loaded */
	MODEL = 'model',
	/** Router mode - server managing multiple model instances */
	ROUTER = 'router'
}

/**
 * Model status enum - matches tools/server/server-models.h from C++ server
 * Used as the `value` field in the status object from /models endpoint
 */
export enum ServerModelStatus {
	UNLOADED = 'unloaded',
	LOADING = 'loading',
	LOADED = 'loaded',
	FAILED = 'failed'
}
