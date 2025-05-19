import { useEffect, useState } from 'react';
import StorageUtils from '../utils/storage';
import { useAppContext } from '../utils/app.context';
import { classNames } from '../utils/misc';
import daisyuiThemes from 'daisyui/theme/object';
import { THEMES } from '../Config';
import {
  Cog8ToothIcon,
  MoonIcon,
  Bars3Icon,
} from '@heroicons/react/24/outline';

export default function Header() {
  const [selectedTheme, setSelectedTheme] = useState(StorageUtils.getTheme());
  const { setShowSettings } = useAppContext();

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
    <div className="flex flex-row items-center pt-6 pb-6 sticky top-0 z-10 bg-base-100">
      {/* open sidebar button */}
      <label htmlFor="toggle-drawer" className="btn btn-ghost lg:hidden">
        <Bars3Icon className="h-5 w-5" />
      </label>

      <div className="grow text-2xl font-bold ml-2">llama.cpp</div>

      {/* action buttons (top right) */}
      <div className="flex items-center">
        <div
          className="tooltip tooltip-bottom"
          data-tip="Settings"
          onClick={() => setShowSettings(true)}
        >
          <button className="btn" aria-hidden={true}>
            {/* settings button */}
            <Cog8ToothIcon className="w-5 h-5" />
          </button>
        </div>

        {/* theme controller is copied from https://daisyui.com/components/theme-controller/ */}
        <div className="tooltip tooltip-bottom" data-tip="Themes">
          <div className="dropdown dropdown-end dropdown-bottom">
            <div tabIndex={0} role="button" className="btn m-1">
              <MoonIcon className="w-5 h-5" />
            </div>
            <ul
              tabIndex={0}
              className="dropdown-content bg-base-300 rounded-box z-[1] w-52 p-2 shadow-2xl h-80 overflow-y-auto"
            >
              <li>
                <button
                  className={classNames({
                    'btn btn-sm btn-block btn-ghost justify-start': true,
                    'btn-active': selectedTheme === 'auto',
                  })}
                  onClick={() => setTheme('auto')}
                >
                  auto
                </button>
              </li>
              {THEMES.map((theme) => (
                <li key={theme}>
                  <input
                    type="radio"
                    name="theme-dropdown"
                    className="theme-controller btn btn-sm btn-block btn-ghost justify-start"
                    aria-label={theme}
                    value={theme}
                    checked={selectedTheme === theme}
                    onChange={(e) => e.target.checked && setTheme(theme)}
                  />
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
