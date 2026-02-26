param(
  [string]$BaseUrl = "http://127.0.0.1:8082",
  [string]$Token = $env:AMEMORIX_DEMO_TOKEN,
  [string]$ConfigPath = "",
  [int]$PollMax = 20,
  [int]$PollIntervalMs = 500,
  [switch]$NoConfigAuto,
  [switch]$CleanBeforeDemo,
  [string]$DemoSource = "acl2026-video-demo",
  [switch]$PauseBetweenSteps,
  [switch]$SkipMemoryAction,
  [switch]$ShowPlan
)

$ErrorActionPreference = "Stop"

function Print-Section([string]$title) {
  Write-Host ""
  Write-Host ("=== {0} ===" -f $title) -ForegroundColor Cyan
}

function Wait-IfNeeded {
  if ($PauseBetweenSteps) {
    [void](Read-Host "Press Enter to continue")
  }
}

function To-Json([object]$obj) {
  return ($obj | ConvertTo-Json -Depth 20)
}

function Resolve-ConfigPath {
  param([string]$UserPath)

  if (-not [string]::IsNullOrWhiteSpace($UserPath)) {
    return $UserPath
  }

  $candidates = @(
    (Join-Path -Path (Get-Location) -ChildPath "config.toml"),
    (Join-Path -Path $PSScriptRoot -ChildPath "..\..\config.toml")
  )

  foreach ($p in $candidates) {
    if (Test-Path $p) {
      return $p
    }
  }
  return ""
}

function Read-ConfigValueFromLine {
  param([string]$Line)
  if ($Line -match "^\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$") {
    return @{
      Key = $matches[1]
      RawValue = $matches[2]
    }
  }
  return $null
}

function Read-DemoConfig {
  param([string]$Path)

  $result = [ordered]@{
    Host = ""
    Port = 8082
    AuthEnabled = $false
    FirstWriteToken = ""
  }

  if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path $Path)) {
    return [PSCustomObject]$result
  }

  $section = ""
  $lines = Get-Content -Path $Path
  foreach ($line in $lines) {
    $noComment = ($line -replace "#.*$", "").Trim()
    if ([string]::IsNullOrWhiteSpace($noComment)) {
      continue
    }
    if ($noComment -match "^\s*\[(.+)\]\s*$") {
      $section = $matches[1].Trim().ToLowerInvariant()
      continue
    }

    $entry = Read-ConfigValueFromLine -Line $noComment
    if ($null -eq $entry) {
      continue
    }

    $key = [string]$entry.Key
    $raw = [string]$entry.RawValue

    if ($section -eq "server") {
      if ($key -eq "host" -and $raw -match '^\s*"(.*)"\s*$') {
        $result.Host = $matches[1]
      }
      elseif ($key -eq "port" -and $raw -match '^\s*(\d+)\s*$') {
        $result.Port = [int]$matches[1]
      }
      continue
    }

    if ($section -eq "auth") {
      if ($key -eq "enabled" -and $raw -match "^\s*(true|false)\s*$") {
        $result.AuthEnabled = ($matches[1].ToLowerInvariant() -eq "true")
      }
      elseif ($key -eq "write_tokens") {
        $tokenMatches = [regex]::Matches($raw, '"([^"]+)"')
        if ($tokenMatches.Count -gt 0) {
          $result.FirstWriteToken = $tokenMatches[0].Groups[1].Value
        }
      }
      continue
    }
  }

  return [PSCustomObject]$result
}

function Read-ErrorBody([System.Management.Automation.ErrorRecord]$err) {
  try {
    $response = $err.Exception.Response
    if ($null -eq $response) {
      return ""
    }
    $stream = $response.GetResponseStream()
    if ($null -eq $stream) {
      return ""
    }
    $reader = New-Object System.IO.StreamReader($stream)
    return $reader.ReadToEnd()
  }
  catch {
    return ""
  }
}

function Invoke-Api {
  param(
    [string]$Method,
    [string]$Path,
    [object]$Body,
    [switch]$AllowFailure,
    [switch]$QuietFailure
  )

  $uri = "{0}{1}" -f $BaseUrl.TrimEnd("/"), $Path
  $headers = @{
    "Content-Type" = "application/json"
  }
  if (-not [string]::IsNullOrWhiteSpace($Token)) {
    $headers["Authorization"] = "Bearer $Token"
  }

  try {
    if ($PSBoundParameters.ContainsKey("Body")) {
      $json = To-Json $Body
      $resp = Invoke-RestMethod -Method $Method -Uri $uri -Headers $headers -Body $json
    }
    else {
      $resp = Invoke-RestMethod -Method $Method -Uri $uri -Headers $headers
    }
    $resp | ConvertTo-Json -Depth 20
    return $resp
  }
  catch {
    $bodyText = Read-ErrorBody $_
    if (-not ($AllowFailure -and $QuietFailure)) {
      Write-Host ("Request failed: {0} {1}" -f $Method, $uri) -ForegroundColor Red
      if (-not [string]::IsNullOrWhiteSpace($bodyText)) {
        Write-Host $bodyText -ForegroundColor Yellow
      }
      if ([string]::IsNullOrWhiteSpace($Token)) {
        Write-Host "Tip: set AMEMORIX_DEMO_TOKEN (or -Token) if write endpoints are protected." -ForegroundColor Yellow
      }
    }
    if ($AllowFailure) {
      return $null
    }
    throw
  }
}

function Wait-Task {
  param(
    [string]$TaskId,
    [string]$TaskKind = "import"
  )

  for ($i = 1; $i -le $PollMax; $i++) {
    Start-Sleep -Milliseconds $PollIntervalMs
    $task = Invoke-Api -Method "GET" -Path ("/v1/{0}/tasks/{1}" -f $TaskKind, $TaskId)
    $status = [string]$task.status
    Write-Host ("poll#{0}: {1}" -f $i, $status) -ForegroundColor DarkGray
    if ($status -in @("succeeded", "failed", "canceled")) {
      return $task
    }
  }
  throw ("Task polling timed out: {0}" -f $TaskId)
}

if ($ShowPlan) {
  Write-Host "S0  (Optional) Clean previous demo data by source"
  Write-Host "S1  Health checks (/healthz, /readyz)"
  Write-Host "S2  Import paragraph task and wait"
  Write-Host "S3  Import relation task and wait"
  Write-Host "S4  Temporal query (/v1/query/time)"
  Write-Host "S5  Memory status before action"
  Write-Host "S6  Memory protect (or skip)"
  Write-Host "S7  Memory status after action"
  Write-Host "S8  Temporal query again"
  exit 0
}

if (-not $NoConfigAuto) {
  $resolvedConfigPath = Resolve-ConfigPath -UserPath $ConfigPath
  $cfg = Read-DemoConfig -Path $resolvedConfigPath

  if (-not $PSBoundParameters.ContainsKey("BaseUrl")) {
    $serverHost = [string]$cfg.Host
    if ([string]::IsNullOrWhiteSpace($serverHost) -or $serverHost -in @("0.0.0.0", "::", "[::]")) {
      $serverHost = "127.0.0.1"
    }
    $BaseUrl = "http://{0}:{1}" -f $serverHost, $cfg.Port
  }

  if (
    -not $PSBoundParameters.ContainsKey("Token") -and
    [string]::IsNullOrWhiteSpace($Token) -and
    -not [string]::IsNullOrWhiteSpace([string]$cfg.FirstWriteToken)
  ) {
    $Token = [string]$cfg.FirstWriteToken
  }

  if (-not [string]::IsNullOrWhiteSpace($resolvedConfigPath)) {
    Write-Host ("Loaded config: {0}" -f $resolvedConfigPath) -ForegroundColor DarkGray
  }
}

Write-Host ("Using BaseUrl: {0}" -f $BaseUrl) -ForegroundColor DarkGray
if ([string]::IsNullOrWhiteSpace($Token)) {
  Write-Host "Token is empty. If auth is enabled for write endpoints, requests may return 401." -ForegroundColor Yellow
}
elseif ($Token -like "*replace-with-your-token*") {
  Write-Host "Token looks like placeholder. Replace auth.write_tokens in config.toml or pass -Token." -ForegroundColor Yellow
}

$now = Get-Date
$eventTime = $now.ToString("yyyy/MM/dd HH:mm")
$timeFrom = $now.AddHours(-2).ToString("yyyy/MM/dd HH:mm")
$timeTo = $now.AddHours(2).ToString("yyyy/MM/dd HH:mm")
$paragraphText = "Alice maintains the ACL demo release checklist and verifies benchmark evidence."
$relationSubject = "Alice"
$relationPredicate = "maintains"
$relationObject = "ACL demo release checklist"
$relationHint = ("{0} {1} {2}" -f $relationSubject, $relationPredicate, $relationObject)

if ($CleanBeforeDemo) {
  Print-Section "S0 / Slide 4: Clean Demo Data"

  $cleanupResp = Invoke-Api -Method "POST" -Path "/api/source/batch_delete" -Body @{ source = $DemoSource } -AllowFailure
  if ($cleanupResp) {
    $cleanupCount = 0
    if ($null -ne $cleanupResp.count) {
      $cleanupCount = [int]$cleanupResp.count
    }
    Write-Host ("source cleanup: source='{0}', deleted_paragraphs={1}" -f $DemoSource, $cleanupCount) -ForegroundColor Green
  }
  else {
    Write-Host "source cleanup skipped or failed; continue demo workflow." -ForegroundColor Yellow
  }

  # Compatibility cleanup for early demo runs where relation might not have been linked to paragraph hash.
  [void](Invoke-Api -Method "POST" -Path "/v1/delete/relation" -Body @{ relation = $relationHint } -AllowFailure -QuietFailure)
  Wait-IfNeeded
}

Print-Section "S1 / Slide 4: Health Checks"
Invoke-Api -Method "GET" -Path "/healthz" | Out-Null
Invoke-Api -Method "GET" -Path "/readyz" | Out-Null
Wait-IfNeeded

Print-Section "S2 / Slide 4: Import Paragraph Task"
$paragraphImportBody = @{
  mode = "paragraph"
  payload = @{
    content = $paragraphText
    source = $DemoSource
    time_meta = @{
      event_time = $eventTime
    }
  }
  options = @{}
}
$paragraphTaskResp = Invoke-Api -Method "POST" -Path "/v1/import/tasks" -Body $paragraphImportBody
$paragraphTaskId = [string]$paragraphTaskResp.task_id
if ([string]::IsNullOrWhiteSpace($paragraphTaskId)) {
  throw "No task_id returned for paragraph import."
}
$paragraphTaskFinal = Wait-Task -TaskId $paragraphTaskId -TaskKind "import"
if ([string]$paragraphTaskFinal.status -ne "succeeded") {
  throw ("Paragraph import did not succeed. status={0}" -f $paragraphTaskFinal.status)
}
$paragraphHash = ""
if ($paragraphTaskFinal.result -and $paragraphTaskFinal.result.hash) {
  $paragraphHash = [string]$paragraphTaskFinal.result.hash
  Write-Host ("Resolved paragraph hash: {0}" -f $paragraphHash) -ForegroundColor Green
}
Wait-IfNeeded

Print-Section "S3 / Slide 4: Import Relation Task"
$relationPayload = @{
  subject = $relationSubject
  predicate = $relationPredicate
  object = $relationObject
  confidence = 0.95
}
if (-not [string]::IsNullOrWhiteSpace($paragraphHash)) {
  # Link relation to paragraph hash so source cleanup can prune relations deterministically.
  $relationPayload["source_paragraph"] = $paragraphHash
}
else {
  $relationPayload["source_paragraph"] = $paragraphText
}
$relationImportBody = @{
  mode = "relation"
  payload = $relationPayload
  options = @{}
}
$relationTaskResp = Invoke-Api -Method "POST" -Path "/v1/import/tasks" -Body $relationImportBody
$relationTaskId = [string]$relationTaskResp.task_id
if ([string]::IsNullOrWhiteSpace($relationTaskId)) {
  throw "No task_id returned for relation import."
}
$relationTaskFinal = Wait-Task -TaskId $relationTaskId -TaskKind "import"
if ([string]$relationTaskFinal.status -ne "succeeded") {
  throw ("Relation import did not succeed. status={0}" -f $relationTaskFinal.status)
}
$relationHash = ""
if ($relationTaskFinal.result -and $relationTaskFinal.result.hash) {
  $relationHash = [string]$relationTaskFinal.result.hash
  Write-Host ("Resolved relation hash: {0}" -f $relationHash) -ForegroundColor Green
}
Wait-IfNeeded

Print-Section "S4 / Slide 4: Temporal Query (Before Memory Action)"
$queryBody = @{
  query = "release checklist"
  time_from = $timeFrom
  time_to = $timeTo
  source = $DemoSource
  top_k = 5
}
$queryBefore = Invoke-Api -Method "POST" -Path "/v1/query/time" -Body $queryBody
Write-Host ("query_before.count = {0}" -f ($queryBefore.count)) -ForegroundColor Green
Wait-IfNeeded

Print-Section "S5 / Slide 4: Memory Status (Before)"
$statusBefore = Invoke-Api -Method "POST" -Path "/v1/memory/status" -Body @{}
$beforeTtl = 0
if ($statusBefore -and $null -ne $statusBefore.ttl_protected_relations) {
  $beforeTtl = [int]$statusBefore.ttl_protected_relations
}
Write-Host ("ttl_protected_relations(before) = {0}" -f $beforeTtl) -ForegroundColor Green
Wait-IfNeeded

if (-not $SkipMemoryAction) {
  Print-Section "S6 / Slide 4: Memory Protect"
  $memoryId = $relationHint
  if (-not [string]::IsNullOrWhiteSpace($relationHash)) {
    $memoryId = $relationHash
  }
  $protectBody = @{
    id = $memoryId
    hours = 24.0
  }
  [void](Invoke-Api -Method "POST" -Path "/v1/memory/protect" -Body $protectBody -AllowFailure)
  Wait-IfNeeded
}
else {
  Print-Section "S6 / Slide 4: Memory Protect Skipped"
  Write-Host "SkipMemoryAction enabled." -ForegroundColor Yellow
  Wait-IfNeeded
}

Print-Section "S7 / Slide 4: Memory Status (After)"
$statusAfter = Invoke-Api -Method "POST" -Path "/v1/memory/status" -Body @{}
$afterTtl = 0
if ($statusAfter -and $null -ne $statusAfter.ttl_protected_relations) {
  $afterTtl = [int]$statusAfter.ttl_protected_relations
}
Write-Host ("ttl_protected_relations(after) = {0}" -f $afterTtl) -ForegroundColor Green
Wait-IfNeeded

Print-Section "S8 / Slide 4: Temporal Query (After Memory Action)"
$queryAfter = Invoke-Api -Method "POST" -Path "/v1/query/time" -Body $queryBody
Write-Host ("query_after.count = {0}" -f ($queryAfter.count)) -ForegroundColor Green
Wait-IfNeeded

Print-Section "Done"
Write-Host "Demo workflow completed. Use benchmark.md and pytest output for Slide 5." -ForegroundColor Green
