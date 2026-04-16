param(
    [string]$OutDir = "dist"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$stageRoot = Join-Path $repoRoot ".build_tmp"
$addonName = "BlenderGIS"
$addonStage = Join-Path $stageRoot $addonName

if (Test-Path $stageRoot) {
    Remove-Item -Recurse -Force $stageRoot
}
New-Item -ItemType Directory -Path $addonStage | Out-Null

$includeFiles = @(
    "__init__.py",
    "geoscene.py",
    "prefs.py",
    "README.md",
    "LICENSE"
)

$includeDirs = @(
    "core",
    "operators",
    "icons",
    "clients"
)

foreach ($f in $includeFiles) {
    $src = Join-Path $repoRoot $f
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $addonStage -Force
    }
}

foreach ($d in $includeDirs) {
    $src = Join-Path $repoRoot $d
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $addonStage -Recurse -Force
    }
}

# Cleanup transient files inside stage.
Get-ChildItem -Path $addonStage -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path $addonStage -Recurse -File -Include "*.pyc","*.pyo" | Remove-Item -Force

$outPath = Join-Path $repoRoot $OutDir
if (-not (Test-Path $outPath)) {
    New-Item -ItemType Directory -Path $outPath | Out-Null
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zipName = "BlenderGIS_UE_Tilemap_$stamp.zip"
$zipPath = Join-Path $outPath $zipName

if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}

Compress-Archive -Path $addonStage -DestinationPath $zipPath -CompressionLevel Optimal

Write-Output $zipPath
