# PyInstaller spec for desktop-solver single-file build.
#
# Usage:
#   pyinstaller desktop-solver.spec
#
# Output:
#   dist/desktop-solver(.exe)
#
# Notes:
# - `console=False` produces a Windows GUI app with no console window.
#   On Linux/macOS the binary runs the same way.
# - `upx=False` because UPX corrupts the native shared libraries shipped
#   inside the llama-cpp-python wheel.
# - `collect_all('llama_cpp')` pulls in the bundled GGML/llama binaries
#   so the .exe can run inference without a separate install.

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules


# llama-cpp-python: native libraries + Python bindings.
llama_datas, llama_binaries, llama_hidden = collect_all("llama_cpp")

# matplotlib: mathtext fonts and rcparams.
mpl_datas = collect_data_files("matplotlib")

# sympy: large enough that PyInstaller's static analysis can miss
# dynamically imported subpackages (parsing, polys, plotting).
sympy_hidden = collect_submodules("sympy")

# antlr4 is pulled in by sympy.parsing.latex; surface it explicitly so
# PyInstaller's tree-shaking doesn't drop it.
extra_hidden = [
    "antlr4",
    "desktop_solver",
    "desktop_solver.gui",
    "desktop_solver.scanner",
    "desktop_solver.chat_store",
    "desktop_solver.solver",
    "desktop_solver.executor",
    "desktop_solver.verifier",
    "desktop_solver.llm_engine",
    "desktop_solver.latex_io",
    "desktop_solver.tag_extractor",
    "desktop_solver.prompts",
    "desktop_solver.cli",
]

a = Analysis(
    ["run.py"],
    pathex=["src"],
    binaries=llama_binaries,
    datas=llama_datas + mpl_datas,
    hiddenimports=llama_hidden + sympy_hidden + extra_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "unittest",
        "pytest",
        "tests",
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="desktop-solver",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
