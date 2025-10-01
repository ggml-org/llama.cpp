export const THINKING_FORMATS = [
	{
		name: 'html',
		startTag: '<think>',
		endTag: '</think>',
		regex: /<think>([\s\S]*?)<\/think>/
	},
	{
		name: 'bracket',
		startTag: '[THINK]',
		endTag: '[/THINK]',
		regex: /\[THINK\]([\s\S]*?)\[\/THINK\]/
	},
	{
		name: 'pipe',
		startTag: '◁think▷',
		endTag: '◁/think▷',
		regex: /◁think▷([\s\S]*?)◁\/think▷/
	}
];
