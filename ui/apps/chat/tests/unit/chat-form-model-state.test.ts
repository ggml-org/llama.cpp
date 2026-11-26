// Guards which model decides the chat form's submit state: the form's own selection
// always wins over the model of the conversation's last assistant message.

import { getChatFormModelState } from '#lib/utils/model-selection.ts';
import { describe, expect, it } from 'vitest';

const AVAILABLE = { id: 'org/available:Q4_K_M', model: 'org/available:Q4_K_M' };
const GONE = 'org/gone:Q4_K_M';

describe('getChatFormModelState', () => {
	it('reports an available model when the form selects one, even if the conversation used a gone model', () => {
		expect(
			getChatFormModelState({
				conversationModel: GONE,
				isRouter: true,
				models: [AVAILABLE],
				selectedModelId: AVAILABLE.id
			})
		).toEqual({ hasModelSelected: true, isSelectedModelAvailable: true });
	});

	it('reports an unavailable model when the form selects one that left the list', () => {
		expect(
			getChatFormModelState({
				conversationModel: AVAILABLE.model,
				isRouter: true,
				models: [AVAILABLE],
				selectedModelId: GONE
			})
		).toEqual({ hasModelSelected: true, isSelectedModelAvailable: false });
	});

	it('falls back to the conversation model when nothing is selected in the form', () => {
		const state = (conversationModel: string) =>
			getChatFormModelState({
				conversationModel,
				isRouter: true,
				models: [AVAILABLE],
				selectedModelId: null
			});

		expect(state(GONE)).toEqual({ hasModelSelected: true, isSelectedModelAvailable: false });
		expect(state(AVAILABLE.model)).toEqual({
			hasModelSelected: true,
			isSelectedModelAvailable: true
		});
	});

	it('has no model selected on a router with neither a selection nor a conversation model', () => {
		expect(
			getChatFormModelState({
				conversationModel: null,
				isRouter: true,
				models: [AVAILABLE],
				selectedModelId: null
			})
		).toEqual({ hasModelSelected: false, isSelectedModelAvailable: false });
	});

	it('always has an available model outside router mode', () => {
		expect(
			getChatFormModelState({
				conversationModel: GONE,
				isRouter: false,
				models: [],
				selectedModelId: null
			})
		).toEqual({ hasModelSelected: true, isSelectedModelAvailable: true });
	});
});
