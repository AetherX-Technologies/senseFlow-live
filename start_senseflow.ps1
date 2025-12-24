$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSCommandPath
Set-Location $root

function Invoke-Compose {
    param([string[]]$ComposeArgs)
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($docker) {
        & docker compose @ComposeArgs
        if ($LASTEXITCODE -ne 0) {
            throw "docker compose failed."
        }
        return
    }
    $compose = Get-Command docker-compose -ErrorAction SilentlyContinue
    if ($compose) {
        & docker-compose @ComposeArgs
        if ($LASTEXITCODE -ne 0) {
            throw "docker-compose failed."
        }
        return
    }
    throw "docker compose not found in PATH."
}

$python = "E:\conda_envs\envs\funasr\python.exe"
if (-not (Test-Path $python)) {
    throw "Python not found: $python"
}

$composeFile = Join-Path $root "docker-compose.mongodb.yml"
if (-not (Test-Path $composeFile)) {
    throw "Missing compose file: $composeFile"
}

Write-Host "[1/3] Starting MongoDB..."
Invoke-Compose @("-f", $composeFile, "-p", "senseflow_mongo", "up", "-d") | Out-Host

$mongoStatus = docker ps --filter "name=senseflow_mongodb" --format "{{.Status}}"
if (-not $mongoStatus) {
    throw "MongoDB container is not running."
}
Write-Host "MongoDB: $mongoStatus"

$existing = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*ws_server.py*" }
if ($existing) {
    Write-Host "[2/3] ws_server already running."
    Write-Host "PID: $($existing.ProcessId)"
    Write-Host "Logs: $root\prototype\logs\ws_server.log"
    Write-Host "[3/3] Ready."
    exit 0
}

$logDir = Join-Path $root "prototype\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$logOut = Join-Path $logDir "ws_server.log"
$logErr = Join-Path $logDir "ws_server.err.log"

if (-not $env:MONGO_URI) { $env:MONGO_URI = "mongodb://localhost:27017" }
if (-not $env:MONGO_DB) { $env:MONGO_DB = "senseflow_live" }
if (-not $env:MONGO_ENABLED) { $env:MONGO_ENABLED = "1" }
$env:PYTHONIOENCODING = "utf-8"

Write-Host "[2/3] Starting ws_server.py..."
Start-Process -FilePath $python -ArgumentList "prototype\ws_server.py" -WorkingDirectory $root -RedirectStandardOutput $logOut -RedirectStandardError $logErr | Out-Null

Start-Sleep -Seconds 2
$running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*ws_server.py*" }
if (-not $running) {
    throw "ws_server failed to start. Check logs: $logOut, $logErr"
}

Write-Host "[3/3] Ready."
Write-Host "WebSocket: ws://localhost:8766"
Write-Host "Logs: $logOut"
