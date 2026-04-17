<#
.SYNOPSIS
    Install EasyPaper Claude Code skills and slash commands (Windows / PowerShell).

.DESCRIPTION
    Sources:
      <repo>\plugins\easypaper\skills\*       -> <target>\skills\<name>\
      <repo>\plugins\easypaper\commands\*.md  -> <target>\commands\<name>.md

    Targets:
      -Global               => $HOME\.claude
      -Project              => (pwd)\.claude
      -ProjectPath <path>   => <path>\.claude

.PARAMETER Global
    Install to $HOME\.claude (user-level skills/commands).

.PARAMETER Project
    Install to (pwd)\.claude.

.PARAMETER ProjectPath
    Install to <path>\.claude. Implies -Project.

.PARAMETER Symlink
    Create symlinks instead of copying. Requires Developer Mode or admin
    privileges on Windows.

.PARAMETER List
    Print what would be installed/removed and exit.

.PARAMETER DryRun
    Show actions without writing.

.PARAMETER Uninstall
    Remove only the items this script would install.

.PARAMETER Yes
    Do not prompt for overwrite confirmation.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_plugin.ps1 -Global

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_plugin.ps1 -Project -DryRun

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_plugin.ps1 -ProjectPath D:\my-research -Symlink

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_plugin.ps1 -Global -Uninstall -Yes
#>

[CmdletBinding()]
param(
    [switch]$Global,
    [switch]$Project,
    [string]$ProjectPath,
    [switch]$Symlink,
    [switch]$List,
    [switch]$DryRun,
    [switch]$Uninstall,
    [Alias('y')]
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'

$PluginDirName = 'easypaper'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir '..')).Path
$PluginRoot = Join-Path $RepoRoot (Join-Path 'plugins' $PluginDirName)
$SkillsSrc   = Join-Path $PluginRoot 'skills'
$CommandsSrc = Join-Path $PluginRoot 'commands'

if (-not (Test-Path $SkillsSrc -PathType Container) -or
    -not (Test-Path $CommandsSrc -PathType Container)) {
    Write-Error "plugin source not found:`n  skills:   $SkillsSrc`n  commands: $CommandsSrc"
    exit 1
}

function Get-SkillDirs   { Get-ChildItem -Path $SkillsSrc   -Directory | Sort-Object Name }
function Get-CommandFiles { Get-ChildItem -Path $CommandsSrc -File -Filter '*.md' | Sort-Object Name }

function Show-SourceListing {
    $skills   = @(Get-SkillDirs)
    $commands = @(Get-CommandFiles)
    Write-Host "Plugin source: $PluginRoot"
    Write-Host ""
    Write-Host ("Skills ({0}):" -f $skills.Count)
    foreach ($d in $skills) { Write-Host "  - $($d.Name)" }
    Write-Host ""
    Write-Host ("Commands ({0}):" -f $commands.Count)
    foreach ($f in $commands) {
        $slash = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
        Write-Host "  - /$slash -> $($f.Name)"
    }
}

if ($List -and -not $Global -and -not $Project -and -not $ProjectPath) {
    Show-SourceListing
    exit 0
}

if ($ProjectPath) { $Project = $true }

if (-not $Global -and -not $Project) {
    Write-Error "must specify -Global or -Project / -ProjectPath <path>"
    exit 2
}
if ($Global -and $Project) {
    Write-Error "-Global and -Project are mutually exclusive"
    exit 2
}

if ($Global) {
    $TargetRoot = Join-Path $HOME '.claude'
    $TargetLabel = "global ($TargetRoot)"
} else {
    if (-not $ProjectPath) { $ProjectPath = (Get-Location).Path }
    if (-not (Test-Path $ProjectPath -PathType Container)) {
        Write-Error "project path does not exist: $ProjectPath"
        exit 1
    }
    $TargetRoot = Join-Path (Resolve-Path $ProjectPath).Path '.claude'
    $TargetLabel = "project ($TargetRoot)"
}

$TargetSkills   = Join-Path $TargetRoot 'skills'
$TargetCommands = Join-Path $TargetRoot 'commands'
$Mode = if ($Symlink) { 'symlink' } else { 'copy' }
$Action = if ($Uninstall) { 'uninstall' } else { 'install' }

# ---- helpers --------------------------------------------------------------

function Invoke-Step {
    param([string]$Description, [scriptblock]$Action)
    if ($DryRun) {
        Write-Host "[dry-run] $Description"
    } else {
        & $Action
    }
}

function Confirm-Action {
    param([string]$Prompt)
    if ($Yes -or $DryRun) { return $true }
    $reply = Read-Host "$Prompt [y/N]"
    return ($reply -match '^[Yy]$')
}

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        Invoke-Step "mkdir $Path" { New-Item -ItemType Directory -Force -Path $Path | Out-Null }
    }
}

function Remove-Target {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Invoke-Step "remove $Path" { Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

function Install-One {
    param(
        [string]$Source,
        [string]$Dest,
        [ValidateSet('Directory', 'File')]
        [string]$Kind
    )
    if (Test-Path -LiteralPath $Dest) {
        Invoke-Step "remove existing $Dest" { Remove-Item -LiteralPath $Dest -Recurse -Force }
    }
    if ($Mode -eq 'symlink') {
        Invoke-Step "symlink $Dest -> $Source" {
            New-Item -ItemType SymbolicLink -Path $Dest -Target $Source | Out-Null
        }
    } else {
        if ($Kind -eq 'Directory') {
            Invoke-Step "copy dir $Source -> $Dest" {
                Copy-Item -LiteralPath $Source -Destination $Dest -Recurse -Force
            }
        } else {
            Invoke-Step "copy file $Source -> $Dest" {
                Copy-Item -LiteralPath $Source -Destination $Dest -Force
            }
        }
    }
}

# ---- enumerate items ------------------------------------------------------

$SkillDirs    = @(Get-SkillDirs)
$CommandFiles = @(Get-CommandFiles)

if ($SkillDirs.Count -eq 0 -and $CommandFiles.Count -eq 0) {
    Write-Error "no skills or commands found under $PluginRoot"
    exit 1
}

# ---- list with target context --------------------------------------------

if ($List) {
    Write-Host "Target: $TargetLabel"
    Write-Host "Mode:   $Mode"
    Write-Host ""
    Write-Host ("Skills ({0}):" -f $SkillDirs.Count)
    foreach ($d in $SkillDirs) {
        Write-Host ("  {0}\skills\{1}\" -f $TargetRoot, $d.Name)
    }
    Write-Host ""
    Write-Host ("Commands ({0}):" -f $CommandFiles.Count)
    foreach ($f in $CommandFiles) {
        $slash = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
        Write-Host ("  {0}\commands\{1}   (slash: /{2})" -f $TargetRoot, $f.Name, $slash)
    }
    exit 0
}

# ---- pre-action banner ---------------------------------------------------

Write-Host "EasyPaper plugin $Action"
Write-Host "  source : $PluginRoot"
Write-Host "  target : $TargetLabel"
Write-Host "  mode   : $Mode"
if ($DryRun) { Write-Host "  dry-run: yes" }
Write-Host ("  skills : {0}`n  commands: {1}`n" -f $SkillDirs.Count, $CommandFiles.Count)

# ---- uninstall -----------------------------------------------------------

if ($Action -eq 'uninstall') {
    if (-not (Confirm-Action "Remove these items from $TargetRoot ?")) {
        Write-Error "aborted"
        exit 1
    }
    foreach ($d in $SkillDirs) {
        Remove-Target (Join-Path $TargetSkills $d.Name)
    }
    foreach ($f in $CommandFiles) {
        Remove-Target (Join-Path $TargetCommands $f.Name)
    }
    Write-Host ""
    Write-Host "Uninstall complete."
    exit 0
}

# ---- install -------------------------------------------------------------

$overwriteCount = 0
foreach ($d in $SkillDirs) {
    if (Test-Path -LiteralPath (Join-Path $TargetSkills $d.Name)) { $overwriteCount++ }
}
foreach ($f in $CommandFiles) {
    if (Test-Path -LiteralPath (Join-Path $TargetCommands $f.Name)) { $overwriteCount++ }
}

if ($overwriteCount -gt 0) {
    if (-not (Confirm-Action "$overwriteCount existing item(s) at the target will be overwritten. Continue?")) {
        Write-Error "aborted"
        exit 1
    }
}

Ensure-Dir $TargetSkills
Ensure-Dir $TargetCommands

foreach ($d in $SkillDirs) {
    $dst = Join-Path $TargetSkills $d.Name
    Install-One -Source $d.FullName -Dest $dst -Kind 'Directory'
    Write-Host "  installed skill   : $($d.Name)"
}

foreach ($f in $CommandFiles) {
    $dst = Join-Path $TargetCommands $f.Name
    Install-One -Source $f.FullName -Dest $dst -Kind 'File'
    $slash = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    Write-Host "  installed command : /$slash"
}

Write-Host ""
Write-Host "Done. Restart Claude Code (or rescan plugins) to pick up changes."
