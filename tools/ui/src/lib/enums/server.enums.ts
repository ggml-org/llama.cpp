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
	SLEEPING = 'sleeping',
	FAILED = 'failed'
}

/**
 * /models/sse event type enum - discriminates the records broadcast on the
 * model status feed in ROUTER mode. Matches the event names emitted by
 * tools/server/server-models.cpp from the C++ server.
 */
export enum ServerModelsSseEventType {
	STATUS_CHANGE = 'status_change',
	MODEL_STATUS = 'model_status',
	STATUS_UPDATE = 'status_update',
	MODELS_RELOAD = 'models_reload',
	MODEL_REMOVE = 'model_remove',
	DOWNLOAD_PROGRESS = 'download_progress'
}
