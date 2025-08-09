import { useEffect, useState } from 'react';
import StorageUtils from '../utils/storage';
import { useAppContext } from '../utils/app.context';
import { classNames } from '../utils/misc';
import daisyuiThemes from 'daisyui/theme/object';
import { THEMES } from '../Config';
import { Cog8ToothIcon, Bars3Icon } from '@heroicons/react/24/outline';

export default function Header() {
  const [selectedTheme, setSelectedTheme] = useState(StorageUtils.getTheme());
  const { serverProps, setShowSettings } = useAppContext();

  const setTheme = (theme: string) => {
    StorageUtils.setTheme(theme);
    setSelectedTheme(theme);
  };

  useEffect(() => {
    document.body.setAttribute('data-theme', selectedTheme);
    document.body.setAttribute(
      'data-color-scheme',
      daisyuiThemes[selectedTheme]?.['color-scheme'] ?? 'auto'
    );
  }, [selectedTheme]);

  return (
    <div className="flex flex-row items-center py-2 sticky top-0 z-10 bg-base-100 border-b border-base-content/10">
      {/* open sidebar button */}
      <label
        htmlFor="toggle-drawer"
        className="btn btn-ghost w-8 h-8 p-0 lg:hidden"
      >
        <Bars3Icon className="h-5 w-5" />
      </label>

      {/* model information*/}
      <div className="grow text-nowrap overflow-hidden truncate ml-2 px-1 sm:px-4 py-0">
        <b>
          {serverProps?.model_alias ||
            serverProps?.model_path
              ?.split(/(\\|\/)/)
              .pop()
              ?.replace(/[-](?:[IQ]+[0-9]+[_\d\w]+)(?:\.[a-z]+)?$/, '')}
        </b>
      </div>

      {/* action buttons (top right) */}
      <div className="flex items-center">
        <div
          className="tooltip tooltip-bottom"
          data-tip="Settings"
          onClick={() => setShowSettings(true)}
        >
          <button
            className="btn btn-ghost w-8 h-8 p-0 rounded-full"
            aria-hidden={true}
          >
            {/* settings button */}
            <Cog8ToothIcon className="w-5 h-5" />
          </button>
        </div>

        {/* theme controller is copied from https://daisyui.com/components/theme-controller/ */}
        <div className="tooltip tooltip-bottom" data-tip="Themes">
          <div className="dropdown dropdown-end dropdown-bottom">
            <div
              tabIndex={0}
              role="button"
              className="btn btn-ghost m-1 w-8 h-8 p-0 rounded-full"
            >
              <div className="bg-base-100 grid shrink-0 grid-cols-2 gap-1 rounded-md p-1 shadow-sm">
                <div className="bg-base-content size-1 rounded-full"></div>{' '}
                <div className="bg-primary size-1 rounded-full"></div>{' '}
                <div className="bg-secondary size-1 rounded-full"></div>{' '}
                <div className="bg-accent size-1 rounded-full"></div>
              </div>
            </div>
            <ul
              tabIndex={0}
              className="dropdown-content rounded-box z-[1] w-50 p-2 shadow-2xl h-80 text-sm overflow-y-auto"
            >
              <li>
                <button
                  className={classNames({
                    'flex gap-3 p-2 btn btn-sm btn-ghost': true,
                    'btn-active': selectedTheme === 'auto',
                  })}
                  onClick={() => setTheme('auto')}
                >
                  <div className="w-32 ml-6 pl-1 truncate text-left">auto</div>
                </button>
              </li>
              {THEMES.map((theme) => (
                <li key={theme}>
                  <button
                    className={classNames({
                      'flex gap-3 p-2 btn btn-sm btn-ghost': true,
                      'btn-active': selectedTheme === theme,
                    })}
                    data-set-theme={theme}
                    data-act-class="[&amp;_svg]:visible"
                    onClick={() => setTheme(theme)}
                  >
                    <div
                      data-theme={theme}
                      className="bg-base-100 grid shrink-0 grid-cols-2 gap-0.5 rounded-md p-1 shadow-sm"
                    >
                      <div className="bg-base-content size-1 rounded-full"></div>{' '}
                      <div className="bg-primary size-1 rounded-full"></div>{' '}
                      <div className="bg-secondary size-1 rounded-full"></div>{' '}
                      <div className="bg-accent size-1 rounded-full"></div>
                    </div>
                    <div className="w-32 truncate text-left">{theme}</div>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
