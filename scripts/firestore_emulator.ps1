# scripts/firestore_emulator.ps1
# Starts the local Firestore emulator (Phase 3 zero-cost dev).
# Requires the gcloud CLI. The emulator does not need a real GCP project.
#
# Usage:
#   .\scripts\firestore_emulator.ps1            # start emulator (keeps running)
#
# Then, in another terminal, run the state test against it:
#   $env:FIRESTORE_EMULATOR = "true"
#   python tests\test_state.py

param(
    [string]$ProjectId = "gyanverse-agent-dev",
    [int]$Port = 8080
)

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: gcloud CLI not found. Install the Google Cloud SDK first:" -ForegroundColor Red
    Write-Host "  https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

$installed = gcloud components list --filter="status=Installed" --format="value(id)" 2>$null
if ($installed -notmatch "cloud-firestore-emulator") {
    Write-Host "Installing cloud-firestore-emulator component..."
    gcloud components install cloud-firestore-emulator --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Component install failed. Retry with:  gcloud components install cloud-firestore-emulator"
        exit 1
    }
}

Write-Host "Starting Firestore emulator on port $Port (project: $ProjectId)..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
gcloud beta emulators firestore start --project=$ProjectId --host-port=localhost:$Port