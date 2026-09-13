<#
.SYNOPSIS
  Enter a Visual Studio x64 developer environment (MSVC + bundled CMake/Ninja) in the current
  PowerShell session.

.DESCRIPTION
  Dot-source this script from the repository root:

      . .\tools\dev-env.ps1
      cmake --preset msvc-release
      cmake --build --preset msvc-release
      ctest --preset msvc-release

  Prefers Visual Studio 2026 (version 18.x) Build Tools, falling back to the newest installation
  that has the C++ toolset. Safe to run repeatedly: it is a no-op when already inside a dev shell.

.PARAMETER VsVersionRange
  vswhere version range to prefer. Default "[18.0,19.0)" (VS 2026).
#>
[CmdletBinding()]
param(
  [string] $VsVersionRange = '[18.0,19.0)'
)

$ErrorActionPreference = 'Stop'

if ($env:VSCMD_VER -and $env:VSCMD_ARG_TGT_ARCH -eq 'x64' -and (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  Write-Host "Already in a Visual Studio $env:VSCMD_VER x64 developer environment."
} else {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found at '$vswhere'. Install Visual Studio Build Tools with the C++ workload."
  }

  $vcComponent = 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
  $installPath = & $vswhere -products * -version $VsVersionRange -requires $vcComponent -latest -property installationPath
  if (-not $installPath) {
    Write-Warning "No Visual Studio in range $VsVersionRange with the C++ toolset; using newest available."
    $installPath = & $vswhere -products * -requires $vcComponent -latest -property installationPath
  }
  if (-not $installPath) {
    throw 'No Visual Studio installation with the MSVC x64 toolset was found.'
  }

  # VsDevCmd invokes vswhere.exe by name; make it resolvable.
  $installerDir = Split-Path $vswhere -Parent
  if (-not (($env:PATH -split ';') -contains $installerDir)) { $env:PATH = "$installerDir;$env:PATH" }

  $devShellDll = Join-Path $installPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
  Import-Module $devShellDll
  Enter-VsDevShell -VsInstallPath $installPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
  Write-Host "Entered developer environment: $installPath"
}

# Ensure the CMake and Ninja bundled with Visual Studio are on PATH (the dev shell normally adds
# them, but only when the corresponding components are installed).
$vsRoot = $env:VSINSTALLDIR
if ($vsRoot) {
  $bundled = @(
    (Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'),
    (Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja')
  )
  foreach ($dir in $bundled) {
    if ((Test-Path $dir) -and -not (($env:PATH -split ';') -contains $dir)) {
      $env:PATH = "$dir;$env:PATH"
    }
  }
}

foreach ($tool in 'cl', 'cmake', 'ninja') {
  $cmd = Get-Command $tool -ErrorAction SilentlyContinue
  if (-not $cmd) { throw "'$tool' is not available after entering the developer environment." }
}
Write-Host ("cmake: " + ((cmake --version) | Select-Object -First 1))
Write-Host ("ninja: " + (ninja --version))
Write-Host ("cl:    " + ((Get-Command cl).Source))
