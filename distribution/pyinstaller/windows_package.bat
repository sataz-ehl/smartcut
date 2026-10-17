@echo off
REM
REM Navigate to the project directory; adjust this if your script is elsewhere
REM cd %~dp0

REM Package the project with PyInstaller (single file)
venv\Scripts\pyinstaller --distpath .\dist --workpath .\build --onefile ^
 -n smartcut -y --hidden-import=uuid smartcut\__main__.py

REM Package the project with PyInstaller (directory output)
venv\Scripts\pyinstaller --distpath .\dist --workpath .\build ^
 -n smartcut -y --hidden-import=uuid smartcut\__main__.py

REM Create zip archive of directory output using 7-Zip
"C:\Program Files\7-Zip\7z.exe" a -tzip ".\dist\smartcut_win.zip" ".\dist\smartcut\*"

REM ..\sign.bat .\dist\smartcut.exe
REM Pause the script to view any messages post-execution
pause
