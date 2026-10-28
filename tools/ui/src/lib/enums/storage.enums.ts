/** Where a persisted preference belongs. */
export enum PreferenceScope {
	/** UI state of this browser only, never synced. */
	DEVICE = 'device',
	/** The user's own data, follows them across browsers once synced. */
	USER = 'user'
}
