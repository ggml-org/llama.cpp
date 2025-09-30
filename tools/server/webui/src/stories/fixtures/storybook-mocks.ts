import { serverStore } from '$lib/stores/server.svelte';

/**
 * Mock server properties for Storybook testing
 * This utility allows setting mock server configurations without polluting production code
 */
export function mockServerProps(props: Partial<ApiLlamaCppServerProps>): void {
	// Directly set the private _serverProps for testing purposes
	(serverStore as unknown as { _serverProps: ApiLlamaCppServerProps })._serverProps = {
		model_path: props.model_path || 'test-model',
		modalities: {
			vision: props.modalities?.vision ?? false,
			audio: props.modalities?.audio ?? false
		},
		...props
	} as ApiLlamaCppServerProps;
}

/**
 * Reset server store to clean state for testing
 */
export function resetServerStore(): void {
	(serverStore as unknown as { _serverProps: ApiLlamaCppServerProps })._serverProps = {
		model_path: '',
		modalities: {
			vision: false,
			audio: false
		}
	} as ApiLlamaCppServerProps;
	(serverStore as unknown as { _error: string })._error = '';
	(serverStore as unknown as { _loading: boolean })._loading = false;
}

/**
 * Common mock configurations for Storybook stories
 */
export const mockConfigs = {
	visionOnly: {
		modalities: { vision: true, audio: false }
	},
	audioOnly: {
		modalities: { vision: false, audio: true }
	},
	bothModalities: {
		modalities: { vision: true, audio: true }
	},
	noModalities: {
		modalities: { vision: false, audio: false }
	}
} as const;
