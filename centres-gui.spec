# PyInstaller spec for centres-gui Windows executable.
# Build with: pyinstaller centres-gui.spec

from PyInstaller.utils.hooks import collect_all

block_cipher = None

all_datas, all_binaries, all_hiddenimports = [], [], []
for pkg in ("PyQt6", "matplotlib", "skimage", "cv2", "networkx", "scipy", "numpy"):
    d, b, h = collect_all(pkg)
    all_datas += d
    all_binaries += b
    all_hiddenimports += h

a = Analysis(
    ["centres/gui.py"],
    pathex=["."],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="centres-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # windowed — no terminal window
    disable_windowed_traceback=False,
    onefile=True,
)
