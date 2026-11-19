// Guards which microphone a recording uses: the system default, unless that is a
// Bluetooth headset and a built-in microphone exists, or the one picked in Settings.

import {
	pickMicrophone,
	VOICE_MICROPHONE_AUTO,
	VOICE_MICROPHONE_DEFAULT
} from '@ggml-org/llama-voice';
import { SETTING_CONFIG_DEFAULT, SETTINGS_KEYS } from '#lib/constants/index.ts';
import { describe, expect, it } from 'vitest';

const input = (deviceId: string, label: string) => ({
	deviceId,
	kind: 'audioinput' as const,
	label
});
const output = (deviceId: string, label: string) => ({
	deviceId,
	kind: 'audiooutput' as const,
	label
});
const BUILT_IN = input('built-in', 'MacBook Pro Microphone (Built-in)');
const AIRPODS = input('airpods', 'AirPods Pro (Bluetooth)');
const USB = input('usb', 'Yeti Stereo Microphone');

describe('pickMicrophone', () => {
	it('defaults to Automatic', () => {
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.VOICE_MICROPHONE]).toBe(VOICE_MICROPHONE_AUTO);
	});

	it('keeps the system default when it is not a Bluetooth headset', () => {
		const devices = [input('default', 'Default - Yeti Stereo Microphone'), BUILT_IN, USB, AIRPODS];

		expect(pickMicrophone(devices, VOICE_MICROPHONE_AUTO)).toBeUndefined();
	});

	it('takes the built-in microphone over a Bluetooth headset that is the default', () => {
		const devices = [
			input('default', 'Default - AirPods Pro (Bluetooth)'),
			AIRPODS,
			BUILT_IN,
			output('airpods-out', 'AirPods Pro (Bluetooth)')
		];

		expect(pickMicrophone(devices, VOICE_MICROPHONE_AUTO)).toBe('built-in');
	});

	it('reads the first input as the default where the browser lists no default entry', () => {
		expect(pickMicrophone([AIRPODS, BUILT_IN], VOICE_MICROPHONE_AUTO)).toBe('built-in');
		expect(pickMicrophone([USB, AIRPODS], VOICE_MICROPHONE_AUTO)).toBeUndefined();
	});

	it('keeps a Bluetooth headset when there is no built-in microphone', () => {
		const devices = [input('default', 'Default - AirPods Pro (Bluetooth)'), AIRPODS, USB];

		expect(pickMicrophone(devices, VOICE_MICROPHONE_AUTO)).toBeUndefined();
	});

	it('uses the system default when asked to', () => {
		expect(pickMicrophone([AIRPODS, BUILT_IN], VOICE_MICROPHONE_DEFAULT)).toBeUndefined();
	});

	it('uses a picked microphone while it is connected', () => {
		expect(pickMicrophone([BUILT_IN, USB], 'usb')).toBe('usb');
		expect(pickMicrophone([BUILT_IN], 'usb')).toBeUndefined();
	});

	it('knows nothing before the browser names the microphones', () => {
		expect(pickMicrophone([input('', ''), input('', '')], VOICE_MICROPHONE_AUTO)).toBeUndefined();
	});
});
