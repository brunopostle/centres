[Setup]
AppId={{6F3A2B1C-D4E5-4F60-8A9B-C0D1E2F34567}
AppName=Centres
AppVersion=0.1.0
AppPublisher=Bruno Postle
AppPublisherURL=https://github.com/brunopostle/centres
AppSupportURL=https://github.com/brunopostle/centres/issues
AppUpdatesURL=https://github.com/brunopostle/centres/releases
DefaultDirName={autopf}\Centres
DefaultGroupName=Centres
AllowNoIcons=yes
LicenseFile=LICENSE
OutputDir=installer
OutputBaseFilename=centres-gui-setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\centres-gui.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Centres"; Filename: "{app}\centres-gui.exe"
Name: "{group}\{cm:UninstallProgram,Centres}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\Centres"; Filename: "{app}\centres-gui.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\centres-gui.exe"; Description: "{cm:LaunchProgram,Centres}"; Flags: nowait postinstall skipifsilent
