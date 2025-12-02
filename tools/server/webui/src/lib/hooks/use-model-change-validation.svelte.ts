import { modelsStore } from '$lib/stores/models.svelte';
import { isRouterMode } from '$lib/stores/server.svelte';
import { toast } from 'svelte-sonner';

interface UseModelChangeValidationOptions {
	/**
	 * Function to get required modalities for validation.
	 * For ChatForm: () => usedModalities() - all messages
	 * For ChatMessageAssistant: () => getModalitiesUpToMessage(messageId) - messages before
	 */
	getRequiredModalities: () => ModelModalities;

	/**
	 * Optional callback to execute after successful validation.
	 * For ChatForm: undefined - just select model
	 * For ChatMessageAssistant: (modelName) => onRegenerate(modelName)
	 */
	onSuccess?: (modelName: string) => void;

	/**
	 * Optional callback for rollback on validation failure.
	 * For ChatForm: (previousId) => selectModelById(previousId)
	 * For ChatMessageAssistant: undefined - no rollback needed
	 */
	onValidationFailure?: (previousModelId: string | null) => Promise<void>;
}

export function useModelChangeValidation(options: UseModelChangeValidationOptions) {
	const { getRequiredModalities, onSuccess, onValidationFailure } = options;

	let previousSelectedModelId: string | null = null;
	const isRouter = $derived(isRouterMode());

	async function handleModelChange(modelId: string, modelName: string): Promise<boolean> {
		try {
			// Store previous selection for potential rollback
			if (onValidationFailure) {
				previousSelectedModelId = modelsStore.selectedModelId;
			}

			// Load model if not already loaded (router mode only)
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

			// Fetch model props to validate modalities
			const props = await modelsStore.fetchModelProps(modelName);

			if (props?.modalities) {
				const requiredModalities = getRequiredModalities();

				// Check if model supports required modalities
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

					// Unload the model if we just loaded it
					if (isRouter && hasLoadedModel) {
						try {
							await modelsStore.unloadModel(modelName);
						} catch (error) {
							console.error('Failed to unload incompatible model:', error);
						}
					}

					// Execute rollback callback if provided
					if (onValidationFailure && previousSelectedModelId) {
						await onValidationFailure(previousSelectedModelId);
					}

					return false;
				}
			}

			// Select the model (validation passed)
			await modelsStore.selectModelById(modelId);

			// Execute success callback if provided
			if (onSuccess) {
				onSuccess(modelName);
			}

			return true;
		} catch (error) {
			console.error('Failed to change model:', error);
			toast.error('Failed to validate model capabilities');

			// Execute rollback callback on error if provided
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
