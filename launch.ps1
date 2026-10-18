param(
    [switch]$SetupOnly,
    [switch]$Offline,
    [switch]$NoBrowser,
    [ValidateRange(1024, 65535)]
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

try {
    if ($env:OS -ne 'Windows_NT') {
        throw 'This launcher requires Windows 10/11 x64.'
    }
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') {
        throw 'This launcher requires an x64 Windows installation.'
    }

    Set-Location -LiteralPath $PSScriptRoot
    $runtimeDirectory = Join-Path $PSScriptRoot '.runtime'
    $uvDirectory = Join-Path $runtimeDirectory 'uv'
    $uvExecutable = Join-Path $uvDirectory 'uv.exe'
    $temporaryDirectory = Join-Path $runtimeDirectory 'temp'
    New-Item -ItemType Directory -Force -Path $uvDirectory, $temporaryDirectory | Out-Null
    $env:UV_CACHE_DIR = Join-Path $runtimeDirectory 'cache'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $runtimeDirectory 'python'
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $PSScriptRoot '.venv'
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONNOUSERSITE = '1'
    $env:TEMP = $temporaryDirectory
    $env:TMP = $temporaryDirectory
    $env:MPLCONFIGDIR = Join-Path $runtimeDirectory 'matplotlib'
    $uvVersion = '0.13.1'

    if (-not (Test-Path -LiteralPath $uvExecutable)) {
        if ($Offline) {
            throw 'First launch needs Internet access. Run run.cmd without -Offline once.'
        }
        Write-Host 'Downloading uv...'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $archivePath = Join-Path $temporaryDirectory 'uv-windows-x64.zip'
        $archiveUrl = "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip"
        $expectedHash = '3b2661a463a5baea7f5076ea2b25ab71d77f36a6c0d809e5350651792f80a672'
        $downloaded = $false
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            try {
                Invoke-WebRequest -UseBasicParsing -Uri $archiveUrl -OutFile $archivePath -TimeoutSec 120
                $hashAlgorithm = [Security.Cryptography.SHA256]::Create()
                $archiveStream = [IO.File]::OpenRead($archivePath)
                try {
                    $actualHash = [BitConverter]::ToString($hashAlgorithm.ComputeHash($archiveStream)).Replace('-', '').ToLowerInvariant()
                }
                finally {
                    $archiveStream.Dispose()
                    $hashAlgorithm.Dispose()
                }
                if ($actualHash -ne $expectedHash) {
                    throw 'The uv archive checksum does not match. Download was not used.'
                }
                $downloaded = $true
                break
            }
            catch {
                if ($attempt -eq 3) { throw }
                Write-Host "Download attempt $attempt failed; retrying..."
            }
        }
        if (-not $downloaded) { throw 'Could not download uv.' }
        [Reflection.Assembly]::LoadWithPartialName('System.IO.Compression.FileSystem') | Out-Null
        [IO.Compression.ZipFile]::ExtractToDirectory($archivePath, $uvDirectory)
        if (-not (Test-Path -LiteralPath $uvExecutable)) {
            throw 'uv.exe was not found in the downloaded archive.'
        }
    }

    $installedVersion = & $uvExecutable --version
    if ($LASTEXITCODE -ne 0 -or $installedVersion -notmatch ('^uv ' + [regex]::Escape($uvVersion) + '(\s|$)')) {
        throw "Unexpected uv version. Expected $uvVersion."
    }

    $pythonVersion = (Get-Content -LiteralPath (Join-Path $PSScriptRoot '.python-version') -Raw).Trim()
    $syncArguments = @('sync', '--locked', '--no-dev', '--python', $pythonVersion)
    if ($Offline) { $syncArguments += '--offline' }
    Write-Host 'Preparing Python and dependencies...'
    & $uvExecutable @syncArguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Dependency setup failed. Check your connection and rerun run.cmd.'
    }
    if ($SetupOnly) {
        Write-Host 'Semgrains is ready. Run run.cmd to open the application.'
        exit 0
    }

    $runArguments = @('run', '--locked', '--no-dev', '--no-sync')
    if ($Offline) { $runArguments += '--offline' }
    $runArguments += @('python', '-m', 'semgrains', '--port', $Port.ToString())
    if ($NoBrowser) { $runArguments += '--no-browser' }
    & $uvExecutable @runArguments
    exit $LASTEXITCODE
}
catch {
    Write-Host ''
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
