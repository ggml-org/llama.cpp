// actions accepted by the realtime inference control endpoint (API_CHAT.CONTROL)
// kept separate from the endpoint paths since these are protocol level verbs
export const CONTROL_ACTION = {
	END_REASONING: 'reasoning_end'
} as const;

export type ControlAction = (typeof CONTROL_ACTION)[keyof typeof CONTROL_ACTION];
