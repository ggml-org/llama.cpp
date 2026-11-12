import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';

/**
 * Mock server properties for Storybook testing
 * This utility allows setting mock server configurations without polluting production code
 */
export function mockServerProps(props: Partial<ApiLlamaCppServerProps>): void {
	// Reset any pointer-events from previous tests (dropdown cleanup)
	const body = document.querySelector('body');

	if (body) body.style.pointerEvents = '';

	// Directly set the props for testing purposes
	(serverStore as unknown as { props: ApiLlamaCppServerProps }).props = {
		modalities: {
			audio: props.modalities?.audio ?? false,
			video: props.modalities?.video ?? false,
			vision: props.modalities?.vision ?? false
		},
		model_path: props.model_path || 'test-model',
		...props
	} as ApiLlamaCppServerProps;

	// Set router mode role so activeModelId can be set
	(serverStore as unknown as { props: ApiLlamaCppServerProps }).props.role = 'ROUTER';

	// Also mock modelsStore methods for modality checking
	const vision = props.modalities?.vision ?? false;
	const audio = props.modalities?.audio ?? false;
	const video = props.modalities?.video ?? false;

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	(modelsStore as any).modelSupportsVision = () => vision;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	(modelsStore as any).modelSupportsAudio = () => audio;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	(modelsStore as any).modelSupportsVideo = () => video;

	// Mock models list with a test model so activeModelId can be resolved
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	(modelsStore as any).models = [
		{
			id: 'test-model',
			model: 'test-model',
			name: 'Test Model'
		}
	];

	// Mock selectedModelId
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	(modelsStore as any).selectedModelId = 'test-model';
}

/**
 * Reset server store to clean state for testing
 */
export function resetServerStore(): void {
	(serverStore as unknown as { props: ApiLlamaCppServerProps }).props = {
		modalities: {
			audio: false,
			video: false,
			vision: false
		},
		model_path: ''
	} as ApiLlamaCppServerProps;
	(serverStore as unknown as { error: string }).error = '';
	(serverStore as unknown as { loading: boolean }).loading = false;
}

/**
 * Common mock configurations for Storybook stories
 */
export const mockConfigs = {
	audioOnly: {
		modalities: { audio: true, vision: false }
	},
	bothModalities: {
		modalities: { audio: true, vision: true }
	},
	noModalities: {
		modalities: { audio: false, video: false, vision: false }
	},
	visionOnly: {
		modalities: { audio: false, vision: true }
	}
} as const;
