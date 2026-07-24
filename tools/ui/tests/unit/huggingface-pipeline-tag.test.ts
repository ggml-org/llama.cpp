import { describe, expect, it } from 'vitest';
import { HuggingFaceService } from '$lib/services/huggingface.service';

describe('pipelineTagLabel / pipelineTagIcon', () => {
	it('returns null for empty input', () => {
		expect(HuggingFaceService.pipelineTagLabel(null)).toBeNull();
		expect(HuggingFaceService.pipelineTagLabel(undefined)).toBeNull();
		expect(HuggingFaceService.pipelineTagLabel('')).toBeNull();
		expect(HuggingFaceService.pipelineTagIcon(null)).toBeNull();
		expect(HuggingFaceService.pipelineTagIcon(undefined)).toBeNull();
		expect(HuggingFaceService.pipelineTagIcon('')).toBeNull();
	});

	it('returns the explicit label when HF_TASKS has an entry', () => {
		expect(HuggingFaceService.pipelineTagLabel('text-generation')).toBe('Text Generation');
		expect(HuggingFaceService.pipelineTagLabel('text2text-generation')).toBe(
			'Text2Text Generation'
		);
		expect(HuggingFaceService.pipelineTagLabel('automatic-speech-recognition')).toBe(
			'Speech Recognition'
		);
		expect(HuggingFaceService.pipelineTagLabel('text-to-speech')).toBe('Text to Speech');
	});

	it('falls back to title-cased kebab-case when entry is absent', () => {
		// `image-text-to-text` is in our list now; use a clearly unknown tag here
		// to assert fallback behaviour.
		expect(HuggingFaceService.pipelineTagLabel('my-broken-tag')).toBe('My-Broken-Tag');
		expect(HuggingFaceService.pipelineTagLabel('image-text-to-text')).toBe('Image-Text-to-Text');
	});

	it('returns a lucide icon name for known pipeline_tags', () => {
		expect(HuggingFaceService.pipelineTagIcon('text-generation')).toBe('message-square');
		expect(HuggingFaceService.pipelineTagIcon('image-text-to-text')).toBe('image-plus');
		expect(HuggingFaceService.pipelineTagIcon('automatic-speech-recognition')).toBe('mic');
		expect(HuggingFaceService.pipelineTagIcon('text-to-speech')).toBe('volume-2');
	});

	it('returns null icon for unknown pipeline_tags so callers can fall back', () => {
		expect(HuggingFaceService.pipelineTagIcon('something-we-do-not-map')).toBeNull();
	});
});
