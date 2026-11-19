/**
 * Which microphone to record from. `auto` keeps the system's default input, except a
 * Bluetooth headset's: recording from its microphone switches the headset into its
 * call mode, which turns what it plays to low-quality mono, so a built-in microphone
 * is used instead when there is one. `default` always takes the system's input; any
 * other value is a device id from enumerateDevices().
 */

import { VOICE_MICROPHONE_CONSTRAINTS } from './constants.js';

export const VOICE_MICROPHONE_AUTO = 'auto';
export const VOICE_MICROPHONE_DEFAULT = 'default';

const BLUETOOTH = /bluetooth|airpods|beats|hands-?free|headset|buds/i;
const BUILT_IN = /built-?in|internal|macbook|imac|laptop/i;
// Chrome lists the system default and the communications device as extra entries
const ALIASES = new Set(['default', 'communications']);

export interface Microphone {
	deviceId: string;
	label: string;
}

/** The microphones the browser lists; their labels are empty until access is granted. */
export async function listMicrophones(): Promise<Microphone[]> {
	if (!navigator.mediaDevices?.enumerateDevices) return [];

	const devices = await navigator.mediaDevices.enumerateDevices();

	return devices
		.filter((device) => device.kind === 'audioinput' && !ALIASES.has(device.deviceId))
		.map((device) => ({ deviceId: device.deviceId, label: device.label }));
}

/**
 * The device id to record from for a choice, or undefined for the system default.
 * `devices` is what enumerateDevices() returned.
 */
export function pickMicrophone(
	devices: readonly Pick<MediaDeviceInfo, 'deviceId' | 'kind' | 'label'>[],
	choice: string
): string | undefined {
	const inputs = devices.filter((device) => device.kind === 'audioinput');

	if (choice === VOICE_MICROPHONE_DEFAULT) return undefined;

	if (choice !== VOICE_MICROPHONE_AUTO) {
		// a microphone that is gone falls back to the default
		return inputs.some((device) => device.deviceId === choice) ? choice : undefined;
	}

	// Chrome names the default entry after the device, e.g. "Default - AirPods (Bluetooth)";
	// elsewhere the first input is the default
	const fallback = inputs.find((device) => device.deviceId === 'default') ?? inputs[0];

	if (!fallback || !BLUETOOTH.test(fallback.label)) return undefined;

	return inputs.find(
		(device) =>
			!ALIASES.has(device.deviceId) && BUILT_IN.test(device.label) && !BLUETOOTH.test(device.label)
	)?.deviceId;
}

/** The device id for a choice, from the microphones listed now. */
export async function resolveMicrophone(choice: string): Promise<string | undefined> {
	if (choice === VOICE_MICROPHONE_DEFAULT || !navigator.mediaDevices?.enumerateDevices) {
		return undefined;
	}

	try {
		return pickMicrophone(await navigator.mediaDevices.enumerateDevices(), choice);
	} catch {
		return undefined;
	}
}

/**
 * What a recording asks the browser for: the raw signal (VOICE_MICROPHONE_CONSTRAINTS)
 * with the caller's processing on top, from `deviceId` when one is picked. The same
 * options always give the same request, so a microphone opened ahead (MicrophoneWarmup)
 * can be told apart from one the recording would ask for differently.
 */
export function microphoneRequest(options: {
	constraints?: MediaTrackConstraints;
	deviceId?: string;
}): MediaStreamConstraints {
	return {
		audio: {
			...VOICE_MICROPHONE_CONSTRAINTS,
			...(options.deviceId ? { deviceId: options.deviceId } : {}),
			...options.constraints
		}
	};
}

/**
 * Whether the page may use the microphone without asking: access was granted before.
 * False where the browser can't tell (no Permissions API, or no `microphone` name in it).
 */
export async function microphoneAllowed(): Promise<boolean> {
	try {
		const status = await navigator.permissions?.query({ name: 'microphone' as PermissionName });

		return status?.state === 'granted';
	} catch {
		return false;
	}
}
