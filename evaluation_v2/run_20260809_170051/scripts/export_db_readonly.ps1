$ErrorActionPreference = 'Stop'

$RunDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $RunDir '..\..')).Path
$LogDir = Join-Path $RunDir 'logs'
$EnvPath = Join-Path $ProjectRoot '.env'

$allowed = @('PGHOST', 'PGPORT', 'PGDB', 'PGUSER', 'PGPASSWORD')
$cfg = @{}
foreach ($line in Get-Content -LiteralPath $EnvPath) {
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
if ($missing) {
    throw "Missing PostgreSQL configuration keys (values suppressed): $($missing -join ', ')"
}

$env:PGHOST = $cfg['PGHOST']
$env:PGPORT = $cfg['PGPORT']
$env:PGDATABASE = $cfg['PGDB']
$env:PGUSER = $cfg['PGUSER']
$env:PGPASSWORD = $cfg['PGPASSWORD']
$env:PGOPTIONS = '-c default_transaction_read_only=on'

$psqlCandidates = @(
    'C:\Program Files\PostgreSQL\18\bin\psql.exe',
    'C:\Program Files\PostgreSQL\18\pgAdmin 4\runtime\psql.exe'
)
$Psql = $psqlCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Psql) {
    throw 'PostgreSQL psql client not found; no dependency will be installed.'
}

function SqlLiteral([string]$value) {
    if ($null -eq $value) { return "''" }
    $clean = $value.Replace(([string][char]0), '').Replace("'", "''")
    return "'$clean'"
}

$queryRows = [System.Collections.Generic.List[object]]::new()
foreach ($row in Import-Csv -LiteralPath (Join-Path $RunDir 'questions.csv')) {
    $queryRows.Add([pscustomobject]@{
        question_id = $row.question_id
        claim_id = ''
        query_type = 'question'
        query_text = $row.question
    })
}
# Claim-level second-pass searches are run reproducibly over the exported
# read-only rag_chunks snapshot by build_corpus.py. Running long claims through
# three unindexed language configurations made the live read-only query exceed
# the operational time limit.

$values = ($queryRows | ForEach-Object {
    '(' + (SqlLiteral $_.question_id) + ',' + (SqlLiteral $_.claim_id) + ',' +
    (SqlLiteral $_.query_type) + ',' + (SqlLiteral $_.query_text) + ')'
}) -join ','

function PsqlPath([string]$path) {
    return $path.Replace('\', '/')
}

$schemaPath = PsqlPath (Join-Path $LogDir 'db_schema.csv')
$chunksPath = PsqlPath (Join-Path $LogDir 'db_chunks.csv')
$profilePath = PsqlPath (Join-Path $LogDir 'db_profile_before.csv')
$lexicalPath = PsqlPath (Join-Path $LogDir 'lexical_pool.csv')
$sqlPath = Join-Path $LogDir 'readonly_export.sql'

$sql = @"
\set ON_ERROR_STOP on
BEGIN READ ONLY;
\copy (SELECT ordinal_position, column_name, data_type, udt_name, is_nullable FROM information_schema.columns WHERE table_schema='public' AND table_name='rag_chunks' ORDER BY ordinal_position) TO '$schemaPath' WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8')
\copy (SELECT doc_id, chunk_id, page_num, topic, source, lang, content, created_at FROM rag_chunks ORDER BY doc_id, chunk_id) TO '$chunksPath' WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8')
\copy (SELECT count(DISTINCT doc_id) AS num_documents, count(*) AS num_chunks, count(*) FILTER (WHERE doc_id IS NULL) AS null_doc_id, count(*) FILTER (WHERE chunk_id IS NULL) AS null_chunk_id, count(*) FILTER (WHERE content IS NULL OR btrim(content)='') AS null_or_empty_content, count(*) FILTER (WHERE topic IS NULL) AS null_topic, count(*) FILTER (WHERE source IS NULL) AS null_source, count(*) FILTER (WHERE lang IS NULL) AS null_lang, count(*) FILTER (WHERE page_num IS NULL) AS null_page_num, count(*) FILTER (WHERE embedding IS NULL) AS null_embedding, min(created_at) AS min_created_at, max(created_at) AS max_created_at FROM rag_chunks) TO '$profilePath' WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8')
\copy (WITH queries(question_id, claim_id, query_type, query_text) AS (VALUES $values), configs(ts_config) AS (VALUES ('simple'),('spanish'),('english')) SELECT q.question_id, q.claim_id, q.query_type, q.query_text, 'lexical_' || cfg.ts_config AS retrieval_method, hit.retrieval_rank, hit.retrieval_score, 'ts_rank_cd' AS score_type, cfg.ts_config, hit.doc_id, hit.chunk_id, hit.page_num, hit.topic, hit.source, hit.lang, hit.content FROM queries q CROSS JOIN configs cfg CROSS JOIN LATERAL (SELECT c.doc_id, c.chunk_id, c.page_num, c.topic, c.source, c.lang, c.content, ts_rank_cd(to_tsvector(cfg.ts_config::regconfig, c.content), plainto_tsquery(cfg.ts_config::regconfig, q.query_text)) AS retrieval_score, row_number() OVER (ORDER BY ts_rank_cd(to_tsvector(cfg.ts_config::regconfig, c.content), plainto_tsquery(cfg.ts_config::regconfig, q.query_text)) DESC, c.doc_id, c.chunk_id) AS retrieval_rank FROM rag_chunks c WHERE to_tsvector(cfg.ts_config::regconfig, c.content) @@ plainto_tsquery(cfg.ts_config::regconfig, q.query_text) ORDER BY retrieval_score DESC, c.doc_id, c.chunk_id LIMIT 100) hit ORDER BY q.question_id, q.query_type, q.claim_id, cfg.ts_config, hit.retrieval_rank) TO '$lexicalPath' WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8')
COMMIT;
"@

[System.IO.File]::WriteAllText($sqlPath, $sql, [System.Text.UTF8Encoding]::new($false))
& $Psql -X -q -v ON_ERROR_STOP=1 -f $sqlPath
if ($LASTEXITCODE -ne 0) {
    throw "Read-only PostgreSQL export failed with exit code $LASTEXITCODE"
}

$summary = [ordered]@{
    status = 'completed'
    transaction_read_only_forced = $true
    psql_version = (& $Psql --version)
    queries = $queryRows.Count
    query_types = @('question')
    fts_configs = @('simple', 'spanish', 'english')
    requested_depth = 100
    embeddings_exported = $false
    secrets_exported = $false
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $LogDir 'db_export_log.json') -Encoding utf8
Write-Output ($summary | ConvertTo-Json -Compress)
