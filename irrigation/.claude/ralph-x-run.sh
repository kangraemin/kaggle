#!/bin/bash
# Ralph-X Auto-generated Loop
# Task: irrigation Kaggle trial & 개선
# Pipeline: Discussion 크롤링+전략수립 → 구현+실행 → 스코어비교 → (3회마다) 제출
# Max iterations: 15

set -euo pipefail

cd /Users/ram/programming/vibecoding/kaggle/irrigation

LOG_FILE=".claude/ralph-x-log.md"
BEST_SCORE_FILE=".claude/ralph-x-best-score.txt"

# Initialize log
cat > "$LOG_FILE" << 'LOGEOF'
# Ralph-X Work Log
Task: irrigation 대회 trial & 개선
Current best val: 0.9853 (trial_002)
Current best public: 0.9609 (trial_002)
LOGEOF

# Ensure best score file
if [ ! -f "$BEST_SCORE_FILE" ]; then
  echo "0.9853" > "$BEST_SCORE_FILE"
fi

# Write stage prompts to temp files
PROMPT_DIR=$(mktemp -d)

cat > "$PROMPT_DIR/stage1.txt" << 'S1EOF'
You are in a Ralph-X loop. This is the STRATEGY stage.

Task: irrigation Kaggle 대회 (playground-series-s6e4) 성능 개선.
Competition URL: https://www.kaggle.com/competitions/playground-series-s6e4

INSTRUCTIONS:
1. Use /browse skill to crawl the discussion page: https://www.kaggle.com/competitions/playground-series-s6e4/discussion
2. Find high-score approaches, useful feature engineering ideas, tricks from top solutions
3. Read .claude/ralph-x-log.md to see what has already been tried
4. Read TRIALS.md to see previous trial history
5. Read the current best trial code in submissions/ to understand the baseline
6. Based on discussion insights + what hasnt been tried, propose ONE specific next strategy
7. Write the strategy to .claude/ralph-x-strategy.md with: approach name, key changes, expected impact
8. Append a summary to .claude/ralph-x-log.md

IMPORTANT: Work autonomously. Do NOT ask questions. Focus on ideas that are DIFFERENT from previous trials.
S1EOF

cat > "$PROMPT_DIR/stage2.txt" << 'S2EOF'
You are in a Ralph-X loop. This is the IMPLEMENTATION stage.

Task: irrigation Kaggle 대회 (playground-series-s6e4) trial 구현 및 실행.

INSTRUCTIONS:
1. Read .claude/ralph-x-strategy.md for the strategy to implement
2. Read .claude/ralph-x-log.md for context on previous work
3. Read TRIALS.md to determine the next trial number
4. Look at submissions/sub_02/trial_002_fe_catboost/trial_002_fe_catboost.py as the base template
5. Create a new submission folder: submissions/sub_XX/trial_NNN_<name>/
6. Write the trial script implementing the strategy from step 1
7. Run the trial script with: python submissions/sub_XX/trial_NNN_<name>/trial_NNN_<name>.py
8. The script MUST save: submission.csv, oof_preds.npy, test_preds.npy, results.json
9. results.json must include "oof_accuracy_ensemble" or "oof_accuracy" as the main val score
10. Append results summary to .claude/ralph-x-log.md
11. Update TRIALS.md with the new trial row

IMPORTANT: Work autonomously. Do NOT ask questions. If the script errors, debug and fix it.
S2EOF

cat > "$PROMPT_DIR/stage3.txt" << 'S3EOF'
You are in a Ralph-X loop. This is the EVALUATION stage.

Task: Compare the latest trial score against the local best.

INSTRUCTIONS:
1. Read .claude/ralph-x-log.md to find the latest trial results
2. Read .claude/ralph-x-best-score.txt for the current best val score
3. Find the latest trial's results.json and extract the main accuracy score
4. Compare: if new score > best score, update .claude/ralph-x-best-score.txt with the new score
5. Append evaluation result to .claude/ralph-x-log.md:
   - "IMPROVED: new_score > old_score" or "NO IMPROVEMENT: new_score <= old_score"
6. If improved, also note which trial is now the best

IMPORTANT: Work autonomously. Do NOT ask questions.
S3EOF

cat > "$PROMPT_DIR/stage4.txt" << 'S4EOF'
You are in a Ralph-X loop. This is the SUBMISSION stage (runs every 3 iterations).

Task: Check if local best was updated recently, and if so submit to Kaggle.

INSTRUCTIONS:
1. Read .claude/ralph-x-log.md to check if any trial in the last 3 iterations improved the best score
2. If YES (best score was updated):
   a. Find the best trial's submission.csv
   b. Submit to Kaggle: kaggle competitions submit -c playground-series-s6e4 -f <path>/submission.csv -m "trial_NNN: brief description"
   c. Wait a moment, then check the score: kaggle competitions submissions -c playground-series-s6e4
   d. Record the public score in .claude/ralph-x-log.md
   e. Update SUBMISSIONS.md with the new submission row
3. If NO (no improvement in last 3 iterations):
   a. Note in log: "No submission - no improvement in last 3 iterations"

IMPORTANT: Work autonomously. Do NOT ask questions. Only submit if there was actual improvement.
S4EOF

MAX_ITER=15

for i in $(seq 1 $MAX_ITER); do
  echo "━━━ Ralph-X iteration $i/$MAX_ITER ━━━"

  # ===== Stage 1: Discussion 크롤링 + 전략 수립 =====
  claude -p "$(cat "$PROMPT_DIR/stage1.txt")"

  # ===== Stage 2: 전략 구현 + 테스트 + 실행 =====
  claude -p "$(cat "$PROMPT_DIR/stage2.txt")"

  # ===== Stage 3: Local best 대비 스코어 비교 =====
  claude -p "$(cat "$PROMPT_DIR/stage3.txt")"

  # ===== Stage 4: 3회마다 Kaggle 제출 (local best 갱신 시) =====
  if [ $((i % 3)) -eq 0 ]; then
    echo "━━━ Iteration $i: Kaggle submission check ━━━"
    claude -p "$(cat "$PROMPT_DIR/stage4.txt")"
  fi

  echo "━━━ Iteration $i complete ━━━"
done

# Cleanup
rm -rf "$PROMPT_DIR"

echo "🏁 Ralph-X finished after $MAX_ITER iterations"
