# PyInstaller spec for centres-gui Windows executable.
# Build with: pyinstaller centres-gui.spec

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Collect only what's genuinely needed rather than collect_all for every package.
# collect_all pulls in test suites, unused backends, and all Qt modules (~100MB+ overhead).
# PyInstaller's built-in cv2 hook handles OpenCV DLLs automatically.
datas = collect_data_files('matplotlib')  # fonts, matplotlibrc, style sheets
binaries = []

hiddenimports = [
    # scipy — only the two submodules actually used
    'scipy.ndimage',
    'scipy.ndimage._ni_support',
    'scipy.spatial',
    'scipy.spatial.distance',
    'scipy._lib.messagestream',
    # skimage — only blob detection
    'skimage.feature',
    'skimage.feature._blob',
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
