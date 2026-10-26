$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create the .venv and install requirements.txt first.'
}
$buildWork = Join-Path $env:TEMP ("object360_build_" + [guid]::NewGuid().ToString('N'))
$stageRoot = Join-Path $PSScriptRoot ('.dist_stage_' + [guid]::NewGuid().ToString('N'))
$finalApp = Join-Path $PSScriptRoot 'dist\Object360Studio'
$stagedApp = Join-Path $stageRoot 'Object360Studio'
& $python -m PyInstaller --noconfirm --clean --onedir --windowed `
    --workpath $buildWork `
    --specpath $buildWork `
    --distpath $stageRoot `
    --name Object360Studio `
    --add-data "$(Join-Path $PSScriptRoot 'yolo11n.pt');." `
    --add-data "$(Join-Path $PSScriptRoot 'fonts\NotoSansThai-Regular.ttf');fonts" `
    --add-data "$(Join-Path $PSScriptRoot 'fonts\NotoSansThai-Bold.ttf');fonts" `
    --add-data "$(Join-Path $PSScriptRoot 'fonts\OFL.txt');fonts" `
    --collect-all ultralytics `
    --collect-all deep_sort_realtime `
    --collect-all customtkinter `
    --collect-submodules torchvision `
    --exclude-module polars `
    --exclude-module matplotlib `
    gui.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

# PyInstaller must not clear dist\Object360Studio: it also contains the user's runs.
$root = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') + '\'
foreach ($path in @($stageRoot, $stagedApp, $finalApp)) {
    if (-not [IO.Path]::GetFullPath($path).StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe build path: $path"
    }
}
$newExe = Join-Path $stagedApp 'Object360Studio.exe'
$newInternal = Join-Path $stagedApp '_internal'
if (-not (Test-Path -LiteralPath $newExe) -or -not (Test-Path -LiteralPath $newInternal)) {
    throw "Incomplete staged build: $stagedApp"
}
if (-not (Test-Path -LiteralPath $finalApp)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $finalApp) | Out-Null
    Move-Item -LiteralPath $stagedApp -Destination $finalApp
} else {
    $running = Get-Process Object360Studio -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -eq (Join-Path $finalApp 'Object360Studio.exe')
    }
    if ($running) { throw 'Close Object360Studio before replacing its files. The staged build is preserved.' }
    $oldExe = Join-Path $finalApp 'Object360Studio.exe'
    $oldInternal = Join-Path $finalApp '_internal'
    $backupExe = Join-Path $stageRoot 'previous.exe'
    $backupInternal = Join-Path $stageRoot 'previous_internal'
    Move-Item -LiteralPath $oldInternal -Destination $backupInternal
    try {
        Move-Item -LiteralPath $newInternal -Destination $oldInternal
        Move-Item -LiteralPath $oldExe -Destination $backupExe
        Move-Item -LiteralPath $newExe -Destination $oldExe
    } catch {
        if (Test-Path -LiteralPath $backupExe) {
            if (Test-Path -LiteralPath $oldExe) {
                Move-Item -LiteralPath $oldExe -Destination (Join-Path $stageRoot 'failed_new.exe')
            }
            Move-Item -LiteralPath $backupExe -Destination $oldExe
        }
        if (Test-Path -LiteralPath $backupInternal) {
            if (Test-Path -LiteralPath $oldInternal) {
                Move-Item -LiteralPath $oldInternal -Destination (Join-Path $stageRoot 'failed_new_internal')
            }
            Move-Item -LiteralPath $backupInternal -Destination $oldInternal
        }
        throw
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $finalApp 'Object360Studio.exe')) -or
    -not (Test-Path -LiteralPath (Join-Path $finalApp '_internal\yolo11n.pt'))) {
    throw 'Installed build is missing the EXE or model; staged files are retained for inspection.'
}
Remove-Item -LiteralPath $stageRoot -Recurse -Force
Write-Host "Ready: $finalApp\Object360Studio.exe"
