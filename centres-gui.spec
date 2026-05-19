# PyInstaller spec for centres-gui Windows executable.
# Build with: pyinstaller centres-gui.spec

import glob
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# collect_data_files covers fonts/matplotlibrc/style sheets for matplotlib.
# collect_all('PyQt6') bundles Qt6 DLLs and platform plugins.
# PyInstaller's built-in cv2 hook handles OpenCV DLLs automatically.
#
# Qt6Core.dll depends on ICU. Two cases:
#   a) PyQt6 ships versioned ICU (icuuc73.dll etc.) in Qt6/bin/ — glob catches them.
#   b) Qt6Core was built against the Windows 10+ system ICU (icuuc.dll in System32) —
#      glob finds nothing, so fall back to System32. Without this, the app crashes
#      under Wine (which doesn't provide the system ICU) with "icuuc.dll not found".
import PyQt6 as _PyQt6
_qt6_bin = os.path.join(os.path.dirname(_PyQt6.__file__), 'Qt6', 'bin')

datas = collect_data_files('matplotlib')  # fonts, matplotlibrc, style sheets
_qt_datas, _qt_binaries, _qt_hidden = collect_all('PyQt6')
datas += _qt_datas

_icu_dlls = [(p, 'PyQt6/Qt6/bin') for p in glob.glob(os.path.join(_qt6_bin, 'icu*.dll'))]
if not _icu_dlls and sys.platform == 'win32':
    _sys32 = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')
    _icu_dlls = [
        (os.path.join(_sys32, n), 'PyQt6/Qt6/bin')
        for n in ('icuuc.dll', 'icudt.dll', 'icuin.dll')
        if os.path.exists(os.path.join(_sys32, n))
    ]

binaries = _qt_binaries + _icu_dlls

hiddenimports = [
    # scipy — only the two submodules actually used
    'scipy.ndimage',
    'scipy.ndimage._ni_support',
    'scipy.spatial',
    'scipy.spatial.distance',
    # skimage — only blob detection
    'skimage.feature',
    'skimage._shared',
    'skimage._shared.utils',
    'skimage.transform',
    # networkx
    'networkx',
    'networkx.algorithms',
    'networkx.classes',
    # matplotlib — Qt/Agg backend only
    'matplotlib.backends.backend_qtagg',
    'matplotlib.backends.backend_agg',
    # PyQt6
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
]

excludes = [
    # unused GUI toolkits
    'tkinter',
    'wx',
    # unused matplotlib backends
    'matplotlib.backends.backend_gtk3agg',
    'matplotlib.backends.backend_gtk3cairo',
    'matplotlib.backends.backend_gtk4agg',
    'matplotlib.backends.backend_gtk4cairo',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_wxagg',
    'matplotlib.backends.backend_wx',
    # unused PyQt6 modules — QtWebEngine alone is ~100MB
    'PyQt6.QtBluetooth',
    'PyQt6.QtDBus',
    'PyQt6.QtDesigner',
    'PyQt6.QtHelp',
    'PyQt6.QtLocation',
    'PyQt6.QtMultimedia',
    'PyQt6.QtMultimediaWidgets',
    'PyQt6.QtNfc',
    'PyQt6.QtPdf',
    'PyQt6.QtPdfWidgets',
    'PyQt6.QtPositioning',
    'PyQt6.QtPrintSupport',
    'PyQt6.QtQml',
    'PyQt6.QtQuick',
    'PyQt6.QtQuick3D',
    'PyQt6.QtRemoteObjects',
    'PyQt6.QtSensors',
    'PyQt6.QtSerialBus',
    'PyQt6.QtSerialPort',
    'PyQt6.QtSql',
    'PyQt6.QtSvg',
    'PyQt6.QtSvgWidgets',
    'PyQt6.QtTest',
    'PyQt6.QtTextToSpeech',
    'PyQt6.QtWebChannel',
    'PyQt6.QtWebEngineCore',
    'PyQt6.QtWebEngineQuick',
    'PyQt6.QtWebEngineWidgets',
    'PyQt6.QtWebSockets',
    'PyQt6.QtXml',
]

a = Analysis(
    ['centres/gui.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
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
    name='centres-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    onefile=True,
)
