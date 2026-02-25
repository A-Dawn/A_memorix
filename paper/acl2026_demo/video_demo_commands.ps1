$ErrorActionPreference = "Stop"

# Usage:
#   1) Start service in another terminal:
#      python -m amemorix serve --config ./config.toml
#   2) Run this script:
#      powershell -ExecutionPolicy Bypass -File .\paper\acl2026_demo\video_demo_commands.ps1

$baseUrl = "http://127.0.0.1:8082"
$token = $env:AMEMORIX_DEMO_TOKEN

if ([string]::IsNullOrWhiteSpace($token)) {
  Write-Host "Set AMEMORIX_DEMO_TOKEN first. Example:" -ForegroundColor Yellow
  Write-Host '$env:AMEMORIX_DEMO_TOKEN = "replace-with-your-token"' -ForegroundColor Yellow
  exit 1
}

$headers = @{
  "Authorization" = "Bearer $token"
  "Content-Type"  = "application/json"
}

function Print-Section([string]$title) {
  Write-Host ""
  Write-Host "=== $title ===" -ForegroundColor Cyan
}

function Json([object]$obj) {
  return ($obj | ConvertTo-Json -Depth 20)
}

Print-Section "Health Checks"
Invoke-RestMethod -Method Get -Uri "$baseUrl/healthz" | ConvertTo-Json
Invoke-RestMethod -Method Get -Uri "$baseUrl/readyz" | ConvertTo-Json

$now = Get-Date
$eventTime = $now.ToString("yyyy/MM/dd HH:mm")
$timeFrom = $now.AddHours(-2).ToString("yyyy/MM/dd HH:mm")
$timeTo = $now.AddHours(2).ToString("yyyy/MM/dd HH:mm")
$demoText = "Alice maintains the ACL demo release checklist and verifies benchmark evidence."

Print-Section "Create Import Task"
$importBody = @{
  mode = "paragraph"
  payload = @{
    content = $demoText
    source = "video-demo"
    time_meta = @{
      event_time = $eventTime
    }
  }
  options = @{}
}

$importResp = Invoke-RestMethod -Method Post -Uri "$baseUrl/v1/import/tasks" -Headers $headers -Body (Json $importBody)
$importResp | ConvertTo-Json -Depth 20

$taskId = $importResp.task_id
if (-not $taskId) {
  Write-Host "No task_id returned. Stop recording and check API output." -ForegroundColor Red
  exit 1
}

Print-Section "Poll Import Task"
$status = ""
for ($i = 0; $i -lt 20; $i++) {
  Start-Sleep -Milliseconds 500
  $taskResp = Invoke-RestMethod -Method Get -Uri "$baseUrl/v1/import/tasks/$taskId" -Headers $headers
  $taskResp | ConvertTo-Json -Depth 20
  $status = [string]$taskResp.status
  if ($status -in @("succeeded", "failed", "canceled")) { break }
}

Print-Section "Temporal Query"
$queryBody = @{
  query = "release checklist"
  time_from = $timeFrom
  time_to = $timeTo
  top_k = 5
}
$queryResp = Invoke-RestMethod -Method Post -Uri "$baseUrl/v1/query/time" -Headers $headers -Body (Json $queryBody)
$queryResp | ConvertTo-Json -Depth 20

Print-Section "Memory Protect (Demo)"
$memoryBody = @{
  id = "release checklist"
}
try {
  $memoryResp = Invoke-RestMethod -Method Post -Uri "$baseUrl/v1/memory/protect" -Headers $headers -Body (Json $memoryBody)
  $memoryResp | ConvertTo-Json -Depth 20
}
catch {
  Write-Host "Memory protect returned non-200; continue recording with explanation." -ForegroundColor Yellow
  Write-Host $_.Exception.Message -ForegroundColor Yellow
}

Print-Section "Temporal Query (After Memory Action)"
$queryResp2 = Invoke-RestMethod -Method Post -Uri "$baseUrl/v1/query/time" -Headers $headers -Body (Json $queryBody)
$queryResp2 | ConvertTo-Json -Depth 20

Print-Section "Done"
Write-Host "Use benchmark.md and pytest output screenshot for the final evidence slide." -ForegroundColor Green
