Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$envExamplePath = Join-Path $projectRoot ".env.example"
$envPath = Join-Path $projectRoot ".env"
$defaultDatabaseUrl = "sqlite:///./reviews.db"

if (-not (Test-Path $envExamplePath)) {
    throw ".env.example was not found at $envExamplePath"
}

if (-not (Test-Path $envPath)) {
    Copy-Item $envExamplePath $envPath
    Write-Host "Created .env from .env.example" -ForegroundColor Green
}

$envContent = Get-Content $envPath
$databasePattern = '^DATABASE_URL='
$databaseLine = "DATABASE_URL=$defaultDatabaseUrl"

if ($envContent -match $databasePattern) {
    $currentDatabaseLine = ($envContent | Where-Object { $_ -match $databasePattern } | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace(($currentDatabaseLine -replace '^DATABASE_URL=', ''))) {
        $envContent = $envContent | ForEach-Object {
            if ($_ -match $databasePattern) { $databaseLine } else { $_ }
        }
        Set-Content -Path $envPath -Value $envContent
        Write-Host "Updated DATABASE_URL in .env to SQLite." -ForegroundColor Green
    }
    else {
        Write-Host "Keeping existing DATABASE_URL from .env: $currentDatabaseLine" -ForegroundColor Yellow
    }
}
else {
    @($databaseLine) + $envContent | Set-Content -Path $envPath
    Write-Host "Added DATABASE_URL to .env." -ForegroundColor Green
}

Write-Host ""
Write-Host "Local setup is ready." -ForegroundColor Green
Write-Host "Next commands:" -ForegroundColor Cyan
Write-Host "1. python -m pip install -r requirements.txt"
Write-Host "2. python scripts/build_artifacts.py --force"
Write-Host "3. uvicorn backend.app.main:app --reload"
Write-Host "4. streamlit run frontend/streamlit_app.py"
Write-Host ""
Write-Host "PowerShell tip:" -ForegroundColor Cyan
Write-Host 'If you ever need a temporary session variable, use: $env:DATABASE_URL = "sqlite:///./reviews.db"'
