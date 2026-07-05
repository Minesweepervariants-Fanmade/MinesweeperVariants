@echo off
setlocal
chcp 65001 >nul

powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -Path '.\minesweepervariants' -Recurse -Include '*.md','*.py' -Exclude 'node_modules','.git','venv' | ForEach-Object { '==== ' + $_.FullName + ' ===='; Get-Content $_.FullName -Raw; '' } | Out-File -FilePath 'project.txt' -Encoding utf8"