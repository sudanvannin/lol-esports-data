# Gold Validation Plan

## Truth Registry

The Gold layer should treat each source as authoritative for a specific slice of the domain instead of trying to make a single source do everything.

| Domain | Source of truth | Why |
| --- | --- | --- |
| Match identity, official schedule, start times, best-of, completion state | Riot LoL Esports official API cached in `data/bronze/matches` | Official competition system used by the project already |
| Game/player box scores, drafts, patch, datacompleteness | Oracle's Elixir CSVs in `data/bronze/oracle_elixir` | Broadest historical competitive dataset with structured game stats |
| League metadata and slugs | Riot LoL Esports official API cached in `data/bronze/leagues` | Official league catalog |
| Tournament metadata | Riot LoL Esports official API cached in `data/bronze/tournaments` | Official tournament catalog |
| Historical fallback and alias resolution | Leaguepedia Cargo schema | Useful for backfilling aliases and cross-checking long-tail historical rows |

## External References

- Riot Developer Portal: https://developer.riotgames.com/docs/lol
- Oracle's Elixir downloads: https://oracleselixir.com/tools/downloads
- Leaguepedia Cargo `MatchSchedule` schema: https://lol.fandom.com/wiki/Module:CargoDeclare/MatchSchedule

## Validation Tiers

### Tier 1: Structural checks

These are hard gates for publishing a Gold snapshot.

- `fact_game_team` must have exactly 2 rows per game.
- `fact_game_player` must have exactly 10 rows per game.
- `fact_game_team` grain `(game_id, team_key)` must be unique.
- `fact_game_player` grain `(game_id, player_key)` must be unique.
- `fact_series` grain `series_key` must be unique.

### Tier 2: Completeness checks

These are warning-level checks that should stay below agreed thresholds.

- Missing `source_team_id` in `fact_game_team`
- Missing `source_player_id` in `fact_game_player`
- Missing `team*_source_team_id` in `fact_series`
- Missing `split_name` in `fact_series`
- Missing `patch_version` in `fact_series`

### Tier 3: Cross-source reconciliation

These checks verify that derived Gold entities still line up with official references.

- Reconcile `fact_series` against Riot official matches using:
  - same match date
  - same league code / mapped official league
  - normalized unordered team pair
  - matching or null-compatible `best_of`
- Track coverage for leagues where the official cache exists.
- Emit unmatched series inside official coverage as `quality_issues`.

### Tier 4: Manual curation backlog

These are not blockers for the first Gold snapshot, but they should feed a future `manual_overrides` table.

- Team alias collisions across eras
- Player alias collisions / name changes
- Leagues with incomplete slug mapping between Oracle codes and Riot slugs
- Series phase labels inferred heuristically rather than sourced officially

## Operational Expectations

- Every Gold snapshot must be immutable and stored under `data/gold/snapshots/<snapshot_id>`.
- Every snapshot must ship with:
  - `manifest.json`
  - `validation_report.json`
  - `dataset_manifest.parquet`
  - `validation_summary.parquet`
  - `quality_issues.parquet`
- `data/gold/latest_snapshot.json` should always point to the latest successful snapshot.

## Recommended Next Validation Expansions

1. Build a `manual_overrides` table with audited fixes for team/player aliases.
2. Backfill official match coverage beyond the currently ingested leagues.
3. Add Leaguepedia-based cross-checks for tournament phase and historical aliases.
4. Add row-hash based change detection to detect source drift between snapshots.
