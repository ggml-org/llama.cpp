import type { ModelOption } from '#lib/types/models.js';

export interface ChatFormModelStateInput {
	conversationModel: string | null;
	isRouter: boolean;
	models: readonly Pick<ModelOption, 'id' | 'model'>[];
	selectedModelId: string | null;
}

export interface ChatFormModelState {
	hasModelSelected: boolean;
	isSelectedModelAvailable: boolean;
}

/**
 * Whether the chat form has a model to send with and whether that model is available.
 * The form's own selection always wins; the conversation's last model only counts
 * when nothing is selected in the form.
 */
export function getChatFormModelState({
	conversationModel,
	isRouter,
	models,
	selectedModelId
}: ChatFormModelStateInput): ChatFormModelState {
	if (!isRouter) return { hasModelSelected: true, isSelectedModelAvailable: true };

	if (selectedModelId) {
		return {
			hasModelSelected: true,
			isSelectedModelAvailable: models.some((option) => option.id === selectedModelId)
		};
	}

	if (conversationModel) {
		return {
			hasModelSelected: true,
			isSelectedModelAvailable: models.some((option) => option.model === conversationModel)
		};
	}

	return { hasModelSelected: false, isSelectedModelAvailable: false };
}
