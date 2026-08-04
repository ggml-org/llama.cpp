# cal-loop-ext plan

## Files to modify
- tools/quantize/tessera/tessera-quantize-db.h  (struct field)
- tools/quantize/tessera/tessera-quantize-db.cpp  (CREATE TABLE col + INSERT col)
- tools/quantize/tessera/test_quantize_db.cpp  (round-trip test)
- tools/tessera/tessera_db.py  (TENSOR_STATS_COLS + insert_tensor_stats)
- tools/tessera/test_tessera_db.py  (recommended_action test)
- tools/tessera/l5_action.py  (NEW: derive_recommended_action rules)
- tools/tessera/calibration_to_tensor_stats.py  (read l5_weights, write recommended_action)
- tools/tessera/test_calibration_to_tensor_stats.py  (test for upsert)
- tools/tessera/calibration_rollup.py  (--l5-outcome join + recommended_action)
- tools/tessera/test_calibration_rollup.py  (l5 outcome test + recommended_action test)
- docs/tessera-unified-db.md  (Phase 13)

## Build
- Need a fresh build dir for the worktree; symlink approach.
