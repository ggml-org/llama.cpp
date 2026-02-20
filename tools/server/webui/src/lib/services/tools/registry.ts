import type {
	ApiToolDefinition,
	SettingsConfigType,
	SettingsConfigValue,
	SettingsFieldConfig
} from '$lib/types';

export type ToolSettingDefinition = SettingsFieldConfig & {
	defaultValue: SettingsConfigValue;
};

export interface ToolRegistration {
	name: string;
	label: string;
	description: string;
	enableConfigKey: string; // key in settings config
	defaultEnabled?: boolean;
	settings?: ToolSettingDefinition[];
	definition: ApiToolDefinition;
	execute: (argsJson: string, config?: SettingsConfigType) => Promise<{ content: string }>;
}

const tools: ToolRegistration[] = [];

export function registerTool(tool: ToolRegistration) {
	const existing = tools.find((t) => t.name === tool.name);
	if (existing) {
		// Allow updates (useful for HMR and incremental tool evolution)
		Object.assign(existing, tool);
		return;
	}
	tools.push(tool);
}

export function getAllTools(): ToolRegistration[] {
	return tools;
}

export function getEnabledToolDefinitions(config: Record<string, unknown>): ApiToolDefinition[] {
	return tools.filter((t) => config[t.enableConfigKey] === true).map((t) => t.definition);
}

export function findToolByName(name: string): ToolRegistration | undefined {
	return tools.find((t) => t.name === name);
}

export function isToolEnabled(name: string, config: Record<string, unknown>): boolean {
	const tool = findToolByName(name);
	return !!(tool && config[tool.enableConfigKey] === true);
}

export function getToolSettingDefaults(): Record<string, boolean> {
	const defaults: Record<string, boolean> = {};
	for (const tool of tools) {
		defaults[tool.enableConfigKey] = tool.defaultEnabled ?? false;
	}
	return defaults;
}

export function getToolConfigDefaults(): Record<string, SettingsConfigValue> {
	const defaults: Record<string, SettingsConfigValue> = {};
	for (const tool of tools) {
		for (const setting of tool.settings ?? []) {
			// Prefer the first registration to avoid non-deterministic overrides
			if (defaults[setting.key] !== undefined) continue;
			defaults[setting.key] = setting.defaultValue;
		}
	}
	return defaults;
}
