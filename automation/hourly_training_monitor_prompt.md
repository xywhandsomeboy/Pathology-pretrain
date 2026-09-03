# CerviPath hourly training check

This invocation is a single scheduled check. Complete the work below once and
exit. Never sleep, poll, watch, wait for another epoch, or schedule another run;
cron is responsible for the next invocation.

Repository:

- `/home/user/90T/xiayw/CerviPath`

Experiment root:

- `/home/user/90T/xiayw/CerviPath/Data/cervical_segmentation_expanded_min64_b4`

## Required procedure

1. Work from current files and processes, not prior conversational memory.
2. Run this deterministic detector exactly once before making a judgment:

   ```bash
   python3 scripts/monitor_cervical_overfitting.py \
     --work-root /home/user/90T/xiayw/CerviPath/Data/cervical_segmentation_expanded_min64_b4 \
     --stop-on-detection --once
   ```

   The detector uses only completed `history.json` epochs, writes
   `overfitting_detected.json` after checkpoints have been saved, and sends
   SIGTERM only to the root trainer whose exact `--output-dir` matches the
   detected run.
3. Inspect all current and queued segmentation runs under `decoder_runs`, their
   complete train/validation histories, exact root processes, checkpoints,
   completion markers, and newly created overfitting markers. Do not infer
   overfitting from an individual batch loss. A running process with no new
   completed validation epoch is a verified wait, not a failure.
4. Treat
   `decoder_runs/distance_only/v1/overfitting_detected.json` as already handled:
   correction commit `1db4f0d` introduced the independent CE tumor weight 1.5
   experiment at `decoder_runs/distance_only_tumor_ce1p5/v1`. Do not modify or
   restart it merely because this old marker exists.
5. A marker is already handled when a sibling `codex_iteration_handled.json`
   records that marker's `detected_at` value. Do not repeat that iteration.

## If there is no new, unhandled overfitting event

- Do not modify code, configuration, checkpoints, manifests, processes, tmux
  sessions, Git state, or GitHub.
- Report the completed epoch metrics and why the evidence does or does not meet
  the detector's criteria, then exit immediately.

## If there is a new, unhandled overfitting event

1. Confirm that the exact matching trainer root process has terminated. Never
   kill DataLoader children, unrelated jobs, or a trainer selected only by a
   broad name match. Preserve `checkpoint_last.pt`, `checkpoint_best.pt`, logs,
   histories, and the detection marker.
2. Diagnose the multi-epoch train/validation divergence and inspect the existing
   correction log in `dinov2_segmentation/OVERFITTING_ITERATIONS.md`.
3. Search the web for primary research papers or official documentation directly
   relevant to the observed pathology-segmentation failure. Prefer a single
   independently testable correction over simultaneous changes. Cite the sources
   in the iteration memo and final report.
4. Implement the smallest evidence-backed correction that addresses the actual
   failure while preserving the model's segmentation objective and the staged
   Stage1/Stage2/decoder training design. Use a new uniquely named output
   directory; never overwrite an experiment.
5. Add or update focused tests, run the relevant test suite, and run a short
   end-to-end smoke test that covers decoder-only, adapters plus decoder, and
   top-backbone joint phases. Verify finite losses and nonzero gradients for the
   intended Stage1, Stage2, and decoder parameter groups.
6. If any test or smoke check fails, do not commit or push and do not launch the
   corrected full experiment. Leave the failure evidence in the final report and
   exit.
7. If verification succeeds, update `OVERFITTING_ITERATIONS.md`, commit only the
   intended tracked source/test/documentation files, and push the current `main`
   branch to `origin`. Never stage `Data/`, checkpoints, training logs, smoke
   artifacts, or the user's untracked `reports/` directory.
8. Start the corrected experiment only after the successful push, only if GPU
   resources permit, in a persistent tmux session with a unique name. Refuse to
   duplicate an already-running command or overwrite existing output.
9. Beside the detection marker, atomically write
   `codex_iteration_handled.json` containing at least the handled `detected_at`,
   commit hash, tests, smoke-test result, corrected output directory, launch
   status, and timestamp. Write this only after successful verification and push.
10. Report exactly what was stopped, changed, tested, committed, pushed, and
    launched, then exit immediately. Do not wait for the new experiment.

Preserve unrelated user work at all times. Do not ask for interactive input in
this scheduled run. If safe progress is impossible, report the exact blocker and
exit without broadening permissions or scope.
