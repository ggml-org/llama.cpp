import { modelsStore } from '$lib/stores/models.svelte';
import { isRouterMode } from '$lib/stores/server.svelte';
import { toast } from 'svelte-sonner';
import type { ModelModalities } from '$lib/types';

interface UseModelChangeValidationOptions {
	/**
	 * Function to get required modalities for validation.
	 */
	getRequiredModalities: () => ModelModalities;

	/**
	 * Optional callback to execute after successful validation.
	 */
	onSuccess?: (modelName: string) => void;

	/**
	 * Optional callback for rollback on validation failure.
	 */
	onValidationFailure?: (previousModelId: string | null) => Promise<void>;
}

export function useModelChangeValidation(options: UseModelChangeValidationOptions) {
	const { getRequiredModalities, onSuccess, onValidationFailure } = options;

	let previousSelectedModelId: string | null = null;
	const isRouter = $derived(isRouterMode());

	async function handleModelChange(modelId: string, modelName: string): Promise<boolean> {
		try {
			if (onValidationFailure) {
				previousSelectedModelId = modelsStore.selectedModelId;
			}

			let hasLoadedModel = false;
			const isModelLoadedBefore = modelsStore.isModelLoaded(modelName);

			if (isRouter && !isModelLoadedBefore) {
				try {
					await modelsStore.loadModel(modelName);
					hasLoadedModel = true;
				} catch {
					toast.error(`Failed to load model "${modelName}"`);
					return false;
				}
			}

			const props = await modelsStore.fetchModelProps(modelName);

			if (props?.modalities) {
				const requiredModalities = getRequiredModalities();

				const missingModalities: string[] = [];
				if (requiredModalities.vision && !props.modalities.vision) {
					missingModalities.push('vision');
				}
				if (requiredModalities.audio && !props.modalities.audio) {
					missingModalities.push('audio');
				}

				if (missingModalities.length > 0) {
					toast.error(
						`Model "${modelName}" doesn't support required modalities: ${missingModalities.join(', ')}. Please select a different model.`
					);

					if (isRouter && hasLoadedModel) {
						try {
							await modelsStore.unloadModel(modelName);
						} catch (error) {
							console.error('Failed to unload incompatible model:', error);
						}
					}

					if (onValidationFailure && previousSelectedModelId) {
						await onValidationFailure(previousSelectedModelId);
					}

					return false;
				}
			}

			await modelsStore.selectModelById(modelId);

			if (onSuccess) {
				onSuccess(modelName);
			}

			return true;
		} catch (error) {
			console.error('Failed to change model:', error);
			toast.error('Failed to validate model capabilities');

			if (onValidationFailure && previousSelectedModelId) {
				await onValidationFailure(previousSelectedModelId);
			}

			return false;
		}
	}

	return {
		handleModelChange
	};
}
