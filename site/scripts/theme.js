/* Tessera Observatory v1 -- theme.
 *
 * FOUC-free dark/light mode. The init script in <head> runs
 * before paint and sets [data-theme] on <html> synchronously.
 * This script runs after paint and wires the toggle button.
 *
 * Priority: stored localStorage > OS preference > default dark
 * (Observatory: dark is the default; light is opt-in).
 */

(function () {
  'use strict';

  var STORAGE_KEY = 'tessera-theme';
  var THEMES = ['dark', 'light'];

  function resolveTheme(stored) {
    if (stored === 'dark' || stored === 'light') return stored;
    try {
      if (window.matchMedia &&
          window.matchMedia('(prefers-color-scheme: light)').matches) {
        return 'light';
      }
    } catch (e) {}
    return 'dark';
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try { document.documentElement.style.colorScheme = theme; } catch (e) {}
  }

  function persistTheme(theme) {
    try { localStorage.setItem(STORAGE_KEY, theme); } catch (e) {}
  }

  function readStored() {
    try { return localStorage.getItem(STORAGE_KEY); } catch (e) { return null; }
  }

  // The init script in <head> has already set data-theme. Read it
  // and apply on this script's load so the toggle button shows
  // the right state.
  var current = document.documentElement.getAttribute('data-theme') || 'dark';

  function setTheme(theme) {
    if (THEMES.indexOf(theme) === -1) return;
    current = theme;
    applyTheme(theme);
    persistTheme(theme);
    var btn = document.querySelector('[data-theme-toggle]');
    if (btn) {
      btn.setAttribute('aria-pressed', theme === 'light' ? 'true' : 'false');
      var label = btn.querySelector('.theme-toggle-label');
      if (label) label.textContent = theme === 'light' ? 'Light' : 'Dark';
    }
  }

  function init() {
    var btn = document.querySelector('[data-theme-toggle]');
    if (!btn) return;
    btn.setAttribute('aria-pressed', current === 'light' ? 'true' : 'false');
    btn.addEventListener('click', function () {
      setTheme(current === 'light' ? 'dark' : 'light');
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
