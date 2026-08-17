# deploy.ps1
# Deploy the GyanVerse Agent to Cloud Run (scales to zero when idle).
# Prereqs: gcloud CLI, authenticated, project set, Cloud Run/Cloud Build/Firestore APIs enabled.
param(
    [string]$ServiceName = "gyanverse-agent",
    [string]$Region = "us-central1",
    [string]$ProjectId = ""
)

$ErrorActionPreference = "Stop"

if (-not $ProjectId) {
    $ProjectId = & gcloud config get-value project 2>$null
    $ProjectId = $ProjectId.Trim()
}
if (-not $ProjectId) {
    Write-Error "Could not determine GCP project. Set -ProjectId or run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
}

# Export the .env values Cloud Run needs (never upload the whole .env file).
$location = "us-central1"
if (Test-Path .env) {
    Get-Content .env | Where-Object { $_ -match "^GOOGLE_CLOUD|^GEMINI_MODEL|^EMBED" } | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $k, $v = $line -split "=", 2
            $envValues += "$k=$v,"
        }
    }
}

Write-Host "Deploying $ServiceName to $ProjectId / $Region ..."
& gcloud run deploy $ServiceName `
    --source . `
    --region $Region `
    --project $ProjectId `
    --allow-unauthenticated `
    --min-instances 0 `
    --set-env-vars "GOOGLE_CLOUD_PROJECT=$ProjectId,$envValues"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Done. Get your URL with:"
    Write-Host "  gcloud run services describe $ServiceName --region $Region --format 'value(status.url)'"
}