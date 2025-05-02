import { useState } from 'react';
import { useAppContext } from '../utils/app.context';
import { CONFIG_DEFAULT, CONFIG_INFO } from '../Config';
import { isDev } from '../Config';
import StorageUtils from '../utils/storage';
import { classNames, isBoolean, isNumeric, isString } from '../utils/misc';
import {
  BeakerIcon,
  ChatBubbleOvalLeftEllipsisIcon,
  Cog6ToothIcon,
  FunnelIcon,
  HandRaisedIcon,
  SquaresPlusIcon,
} from '@heroicons/react/24/outline';
import { OpenInNewTab } from '../utils/common';

type SettKey = keyof typeof CONFIG_DEFAULT;

const BASIC_KEYS: SettKey[] = [
  'temperature',
  'top_k',
  'top_p',
  'min_p',
  'max_tokens',
];
const SAMPLER_KEYS: SettKey[] = [
  'dynatemp_range',
  'dynatemp_exponent',
  'typical_p',
  'xtc_probability',
  'xtc_threshold',
];
const PENALTY_KEYS: SettKey[] = [
  'repeat_last_n',
  'repeat_penalty',
  'presence_penalty',
  'frequency_penalty',
  'dry_multiplier',
  'dry_base',
  'dry_allowed_length',
  'dry_penalty_last_n',
];

enum SettingInputType {
  SHORT_INPUT,
  LONG_INPUT,
  CHECKBOX,
  CUSTOM,
}

interface SettingFieldInput {
  type: Exclude<SettingInputType, SettingInputType.CUSTOM>;
  label: string | React.ReactElement;
  help?: string | React.ReactElement;
  key: SettKey;
}

interface SettingFieldCustom {
  type: SettingInputType.CUSTOM;
  key: SettKey;
  component:
    | string
    | React.FC<{
        value: string | boolean | number;
        onChange: (value: string) => void;
      }>;
}

interface SettingSection {
  title: React.ReactElement;
  fields: (SettingFieldInput | SettingFieldCustom)[];
}

const ICON_CLASSNAME = 'w-4 h-4 mr-1 inline';

const SETTING_SECTIONS: SettingSection[] = [
  {
    title: (
      <>
        <Cog6ToothIcon className={ICON_CLASSNAME} />
        General
      </>
    ),
    fields: [
      {
        type: SettingInputType.SHORT_INPUT,
        label: 'API Key',
        key: 'apiKey',
      },
      {
        type: SettingInputType.LONG_INPUT,
        label: 'System Message (will be disabled if left empty)',
        key: 'systemMessage',
      },
      ...BASIC_KEYS.map(
        (key) =>
          ({
            type: SettingInputType.SHORT_INPUT,
            label: key,
            key,
          }) as SettingFieldInput
      ),
    ],
  },
  {
    title: (
      <>
        <FunnelIcon className={ICON_CLASSNAME} />
        Samplers
      </>
    ),
    fields: [
      {
        type: SettingInputType.SHORT_INPUT,
        label: 'Samplers queue',
        key: 'samplers',
      },
      ...SAMPLER_KEYS.map(
        (key) =>
          ({
            type: SettingInputType.SHORT_INPUT,
            label: key,
            key,
          }) as SettingFieldInput
      ),
    ],
  },
  {
    title: (
      <>
        <HandRaisedIcon className={ICON_CLASSNAME} />
        Penalties
      </>
    ),
    fields: PENALTY_KEYS.map((key) => ({
      type: SettingInputType.SHORT_INPUT,
      label: key,
      key,
    })),
  },
  {
    title: (
      <>
        <ChatBubbleOvalLeftEllipsisIcon className={ICON_CLASSNAME} />
        Reasoning
      </>
    ),
    fields: [
      {
        type: SettingInputType.CHECKBOX,
        label: 'Expand thought process by default when generating messages',
        key: 'showThoughtInProgress',
      },
      {
        type: SettingInputType.CHECKBOX,
        label:
          'Exclude thought process when sending requests to API (Recommended for DeepSeek-R1)',
        key: 'excludeThoughtOnReq',
      },
    ],
  },
  {
    title: (
      <>
        <SquaresPlusIcon className={ICON_CLASSNAME} />
        Advanced
      </>
    ),
    fields: [
      {
        type: SettingInputType.CUSTOM,
        key: 'custom', // dummy key, won't be used
        component: () => {
          const debugImportDemoConv = async () => {
            const res = await fetch('/demo-conversation.json');
            const demoConv = await res.json();
            StorageUtils.remove(demoConv.id);
            for (const msg of demoConv.messages) {
              StorageUtils.appendMsg(demoConv.id, msg);
            }
          };
          return (
            <button className="btn" onClick={debugImportDemoConv}>
              (debug) Import demo conversation
            </button>
          );
        },
      },
      {
        type: SettingInputType.CHECKBOX,
        label: 'Show tokens per second',
        key: 'showTokensPerSecond',
      },
      {
        type: SettingInputType.LONG_INPUT,
        label: (
          <>
            Custom JSON config (For more info, refer to{' '}
            <OpenInNewTab href="https://github.com/ggerganov/llama.cpp/blob/master/tools/server/README.md">
              server documentation
            </OpenInNewTab>
            )
          </>
        ),
        key: 'custom',
      },
    ],
  },
  {
    title: (
      <>
        <BeakerIcon className={ICON_CLASSNAME} />
        Experimental
      </>
    ),
    fields: [
      {
        type: SettingInputType.CUSTOM,
        key: 'custom', // dummy key, won't be used
        component: () => (
          <>
            <p className="mb-8">
              Experimental features are not guaranteed to work correctly.
              <br />
              <br />
              If you encounter any problems, create a{' '}
              <OpenInNewTab href="https://github.com/ggerganov/llama.cpp/issues/new?template=019-bug-misc.yml">
                Bug (misc.)
              </OpenInNewTab>{' '}
              report on Github. Please also specify <b>webui/experimental</b> on
              the report title and include screenshots.
              <br />
              <br />
              Some features may require packages downloaded from CDN, so they
              need internet connection.
            </p>
          </>
        ),
      },
      {
        type: SettingInputType.CHECKBOX,
        label: (
          <>
            <b>Enable Python interpreter</b>
            <br />
            <small className="text-xs">
              This feature uses{' '}
              <OpenInNewTab href="https://pyodide.org">pyodide</OpenInNewTab>,
              downloaded from CDN. To use this feature, ask the LLM to generate
              Python code inside a Markdown code block. You will see a "Run"
              button on the code block, near the "Copy" button.
            </small>
          </>
        ),
        key: 'pyIntepreterEnabled',
      },
    ],
  },
];

export default function SettingDialog({
  show,
  onClose,
}: {
  show: boolean;
  onClose: () => void;
}) {
  const { config, saveConfig } = useAppContext();
  const [sectionIdx, setSectionIdx] = useState(0);

  // clone the config object to prevent direct mutation
  const [localConfig, setLocalConfig] = useState<typeof CONFIG_DEFAULT>(
    JSON.parse(JSON.stringify(config))
  );

  const resetConfig = () => {
    if (window.confirm('Are you sure you want to reset all settings?')) {
      setLocalConfig(CONFIG_DEFAULT);
    }
  };

  const handleSave = () => {
    // copy the local config to prevent direct mutation
    const newConfig: typeof CONFIG_DEFAULT = JSON.parse(
      JSON.stringify(localConfig)
    );
    // validate the config
    for (const key in newConfig) {
      const value = newConfig[key as SettKey];
      const mustBeBoolean = isBoolean(CONFIG_DEFAULT[key as SettKey]);
      const mustBeString = isString(CONFIG_DEFAULT[key as SettKey]);
      const mustBeNumeric = isNumeric(CONFIG_DEFAULT[key as SettKey]);
      if (mustBeString) {
        if (!isString(value)) {
          alert(`Value for ${key} must be string`);
          return;
        }
      } else if (mustBeNumeric) {
        const trimmedValue = value.toString().trim();
        const numVal = Number(trimmedValue);
        if (isNaN(numVal) || !isNumeric(numVal) || trimmedValue.length === 0) {
          alert(`Value for ${key} must be numeric`);
          return;
        }
        // force conversion to number
        // @ts-expect-error this is safe
        newConfig[key] = numVal;
      } else if (mustBeBoolean) {
        if (!isBoolean(value)) {
          alert(`Value for ${key} must be boolean`);
          return;
        }
      } else {
        console.error(`Unknown default type for key ${key}`);
      }
    }
    if (isDev) console.log('Saving config', newConfig);
    saveConfig(newConfig);
    onClose();
  };

  const onChange = (key: SettKey) => (value: string | boolean) => {
    // note: we do not perform validation here, because we may get incomplete value as user is still typing it
    setLocalConfig({ ...localConfig, [key]: value });
  };

  return (
    <dialog className={classNames({ modal: true, 'modal-open': show })}>
      <div className="modal-box w-11/12 max-w-3xl">
        <h3 className="text-lg font-bold mb-6">Settings</h3>
        <div className="flex flex-col md:flex-row h-[calc(90vh-12rem)]">
          {/* Left panel, showing sections - Desktop version */}
          <div className="hidden md:flex flex-col items-stretch pr-4 mr-4 border-r-2 border-base-200">
            {SETTING_SECTIONS.map((section, idx) => (
              <div
                key={idx}
                className={classNames({
                  'btn btn-ghost justify-start font-normal w-44 mb-1': true,
                  'btn-active': sectionIdx === idx,
                })}
                onClick={() => setSectionIdx(idx)}
                dir="auto"
              >
                {section.title}
              </div>
            ))}
          </div>

          {/* Left panel, showing sections - Mobile version */}
          <div className="md:hidden flex flex-row gap-2 mb-4">
            <details className="dropdown">
              <summary className="btn bt-sm w-full m-1">
                {SETTING_SECTIONS[sectionIdx].title}
              </summary>
              <ul className="menu dropdown-content bg-base-100 rounded-box z-[1] w-52 p-2 shadow">
                {SETTING_SECTIONS.map((section, idx) => (
                  <div
                    key={idx}
                    className={classNames({
                      'btn btn-ghost justify-start font-normal': true,
                      'btn-active': sectionIdx === idx,
                    })}
                    onClick={() => setSectionIdx(idx)}
                    dir="auto"
                  >
                    {section.title}
                  </div>
                ))}
              </ul>
            </details>
          </div>

          {/* Right panel, showing setting fields */}
          <div className="grow overflow-y-auto px-4">
            {SETTING_SECTIONS[sectionIdx].fields.map((field, idx) => {
              const key = `${sectionIdx}-${idx}`;
              if (field.type === SettingInputType.SHORT_INPUT) {
                return (
                  <SettingsModalShortInput
                    key={key}
                    configKey={field.key}
                    value={localConfig[field.key]}
                    onChange={onChange(field.key)}
                    label={field.label as string}
                  />
                );
              } else if (field.type === SettingInputType.LONG_INPUT) {
                return (
                  <SettingsModalLongInput
                    key={key}
                    configKey={field.key}
                    value={localConfig[field.key].toString()}
                    onChange={onChange(field.key)}
                    label={field.label as string}
                  />
                );
              } else if (field.type === SettingInputType.CHECKBOX) {
                return (
                  <SettingsModalCheckbox
                    key={key}
                    configKey={field.key}
                    value={!!localConfig[field.key]}
                    onChange={onChange(field.key)}
                    label={field.label as string}
                  />
                );
              } else if (field.type === SettingInputType.CUSTOM) {
                return (
                  <div key={key} className="mb-2">
                    {typeof field.component === 'string'
                      ? field.component
                      : field.component({
                          value: localConfig[field.key],
                          onChange: onChange(field.key),
                        })}
                  </div>
                );
              }
            })}

            <p className="opacity-40 mb-6 text-sm mt-8">
              Settings are saved in browser's localStorage
            </p>
          </div>
        </div>

        <div className="modal-action">
          <button className="btn" onClick={resetConfig}>
            Reset to default
          </button>
          <button className="btn" onClick={onClose}>
            Close
          </button>
          <button className="btn btn-primary" onClick={handleSave}>
            Save
          </button>
        </div>
      </div>
    </dialog>
  );
}

function SettingsModalLongInput({
  configKey,
  value,
  onChange,
  label,
}: {
  configKey: SettKey;
  value: string;
  onChange: (value: string) => void;
  label?: string;
}) {
  return (
    <label className="form-control mb-2">
      <div className="label inline">{label || configKey}</div>
      <textarea
        className="textarea textarea-bordered h-24"
        placeholder={`Default: ${CONFIG_DEFAULT[configKey] || 'none'}`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

function SettingsModalShortInput({
  configKey,
  value,
  onChange,
  label,
}: {
  configKey: SettKey;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  value: any;
  onChange: (value: string) => void;
  label?: string;
}) {
  const helpMsg = CONFIG_INFO[configKey];

  return (
    <>
      {/* on mobile, we simply show the help message here */}
      {helpMsg && (
        <div className="block md:hidden mb-1">
          <b>{label || configKey}</b>
          <br />
          <p className="text-xs">{helpMsg}</p>
        </div>
      )}
      <label className="input input-bordered join-item grow flex items-center gap-2 mb-2">
        <div className="dropdown dropdown-hover">
          <div tabIndex={0} role="button" className="font-bold hidden md:block">
            {label || configKey}
          </div>
          {helpMsg && (
            <div className="dropdown-content menu bg-base-100 rounded-box z-10 w-64 p-2 shadow mt-4">
              {helpMsg}
            </div>
          )}
        </div>
        <input
          type="text"
          className="grow"
          placeholder={`Default: ${CONFIG_DEFAULT[configKey] || 'none'}`}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      </label>
    </>
  );
}

function SettingsModalCheckbox({
  configKey,
  value,
  onChange,
  label,
}: {
  configKey: SettKey;
  value: boolean;
  onChange: (value: boolean) => void;
  label: string;
}) {
  return (
    <div className="flex flex-row items-center mb-2">
      <input
        type="checkbox"
        className="toggle"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="ml-4">{label || configKey}</span>
    </div>
  );
}
