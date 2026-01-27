/**
 * BaseClient - Abstract base class for client classes
 *
 * Provides common store callback management pattern used by all clients.
 * Clients extend this class to inherit the store callback infrastructure.
 *
 * **Usage:**
 * ```typescript
 * interface MyStoreCallbacks {
 *   setSomeState: (value: string) => void;
 * }
 *
 * class MyClient extends BaseClient<MyStoreCallbacks> {
 *   doSomething() {
 *     this.store.setSomeState('value');
 *   }
 * }
 * ```
 */
export abstract class BaseClient<TCallbacks> {
	private _storeCallbacks: TCallbacks | null = null;

	/**
	 * Sets callbacks for store state updates.
	 * Called by the corresponding store during initialization.
	 */
	setStoreCallbacks(callbacks: TCallbacks): void {
		this._storeCallbacks = callbacks;
	}

	/**
	 * Gets the store callbacks, throwing if not initialized.
	 * Use this in derived classes to access store callbacks safely.
	 */
	protected get store(): TCallbacks {
		if (!this._storeCallbacks) {
			throw new Error(`${this.constructor.name}: Store callbacks not initialized`);
		}
		return this._storeCallbacks;
	}

	/**
	 * Checks if store callbacks have been initialized.
	 */
	protected get hasStoreCallbacks(): boolean {
		return this._storeCallbacks !== null;
	}
}
