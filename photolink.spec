# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=['C:\\Users\\choph\\photomatcher\\env\\Scripts', 'C:\\Users\\choph\\photomatcher'],
    binaries=[],
    datas=[('./config.ini', '.'), ('./src/photolink/pipeline/', 'src/photolink/pipeline/'), ('./assets', 'assets'), ('./env', 'env')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [('C:/Users/choph/photomatcher/env/Scripts/python3.exe', None, 'OPTION')],
    exclude_binaries=True,
    name='photolink',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/img/logo.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='photolink',
)
