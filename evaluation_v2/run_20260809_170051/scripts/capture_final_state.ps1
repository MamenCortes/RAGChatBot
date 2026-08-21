$ErrorActionPreference = 'Stop'

$RunDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $RunDir '..\..')).Path
$LogDir = Join-Path $RunDir 'logs'
$allowed = @('PGHOST', 'PGPORT', 'PGDB', 'PGUSER', 'PGPASSWORD')
$cfg = @{}
foreach ($line in Get-Content -LiteralPath (Join-Path $ProjectRoot '.env')) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
        $key = $matches[1]
        if ($allowed -contains $key) {
            $value = $matches[2].Trim()
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            $cfg[$key] = $value
        }
    }
}
$missing = $allowed | Where-Object { -not $cfg.ContainsKey($_) }
if ($missing) { throw "Missing PostgreSQL configuration keys (values suppressed): $($missing -join ', ')" }

$env:PGHOST = $cfg['PGHOST']
$env:PGPORT = $cfg['PGPORT']
$env:PGDATABASE = $cfg['PGDB']
$env:PGUSER = $cfg['PGUSER']
$env:PGPASSWORD = $cfg['PGPASSWORD']
$env:PGOPTIONS = '-c default_transaction_read_only=on'
$Psql = 'C:\Program Files\PostgreSQL\18\bin\psql.exe'
if (-not (Test-Path -LiteralPath $Psql)) { throw 'psql not found; no installation attempted.' }

$profilePath = (Join-Path $LogDir 'db_profile_after.csv').Replace('\', '/')
$sqlPath = Join-Path $LogDir 'readonly_final_check.sql'
$sql = @"
\set ON_ERROR_STOP on
BEGIN READ ONLY;
\copy (SELECT count(DISTINCT doc_id) AS num_documents, count(*) AS num_chunks, count(*) FILTER (WHERE doc_id IS NULL) AS null_doc_id, count(*) FILTER (WHERE chunk_id IS NULL) AS null_chunk_id, count(*) FILTER (WHERE content IS NULL OR btrim(content)='') AS null_or_empty_content, count(*) FILTER (WHERE topic IS NULL) AS null_topic, count(*) FILTER (WHERE source IS NULL) AS null_source, count(*) FILTER (WHERE lang IS NULL) AS null_lang, count(*) FILTER (WHERE page_num IS NULL) AS null_page_num, count(*) FILTER (WHERE embedding IS NULL) AS null_embedding, min(created_at) AS min_created_at, max(created_at) AS max_created_at FROM rag_chunks) TO '$profilePath' WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8')
COMMIT;
"@
[System.IO.File]::WriteAllText($sqlPath, $sql, [System.Text.UTF8Encoding]::new($false))
& $Psql -X -q -v ON_ERROR_STOP=1 -f $sqlPath
if ($LASTEXITCODE -ne 0) { throw "Final read-only profile failed with exit code $LASTEXITCODE" }

$gitLines = git -C $ProjectRoot status --short --untracked-files=all
[System.IO.File]::WriteAllLines((Join-Path $LogDir 'git_status_final.txt'), [string[]]$gitLines, [System.Text.UTF8Encoding]::new($false))
Write-Output '{"database_profile_captured":true,"git_status_captured":true,"transaction_read_only_forced":true}'
