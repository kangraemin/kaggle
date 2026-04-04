#!/bin/bash
# Ralph-X Auto-generated Loop
# Task: irrigation 대회 — discussion 분석 → 전략 탐색 → 개발 → 검증 → 제출
# Pipeline: Discussion 리서치 → 전략+개발 → 검증+제출판단
# Max iterations: 15

set -euo pipefail

PROJECT_DIR="/Users/ram/programming/vibecoding/kaggle/irrigation"
LOG_FILE="$PROJECT_DIR/.claude/ralph-x-log.md"
CHECKLIST_FILE="$PROJECT_DIR/.claude/ralph-x-checklist.md"
BEST_SCORE_FILE="$PROJECT_DIR/.claude/ralph-x-best-score.txt"
COMPETITION="playground-series-s6e4"

mkdir -p "$PROJECT_DIR/.claude"

# Initialize log
cat > "$LOG_FILE" << 'LOGEOF'
# Ralph-X Work Log
Task: irrigation 대회 전략 탐색 + 자동 제출
Started: $(date '+%Y-%m-%d %H:%M')
Current best public: 0.96085 (trial_002_fe_catboost)
LOGEOF

# Initialize checklist
cat > "$CHECKLIST_FILE" << 'CHECKEOF'
# Completion Checklist
- [ ] 12개 이상 전략 시도
CHECKEOF

# Track best OOF score
echo "0.9853" > "$BEST_SCORE_FILE"

MAX_ITER=15
TRIALS_DONE=0

# ━━━ Stage 0: Discussion Research (1회만) ━━━
echo "━━━ Ralph-X Stage 0: Discussion Research ━━━"

claude -p "$(cat << 'PROMPTEOF'
You are in a Ralph-X loop. Stage 0: Discussion Research.

Working directory: /Users/ram/programming/vibecoding/kaggle/irrigation

Task: Kaggle playground-series-s6e4 (Irrigation Prediction) discussion 분석.
Competition URL: https://www.kaggle.com/competitions/playground-series-s6e4/discussion

Use /browse skill to crawl the Kaggle discussion page for this competition.
Find top-voted discussions and extract useful strategies, feature engineering ideas, model choices, and insights.

IMPORTANT:
- Use the browse tool to navigate to the discussion page
- Extract at least 5-10 key insights from discussions
- Focus on: feature engineering, model selection, ensemble strategies, data augmentation, validation schemes
- Write all findings to /Users/ram/programming/vibecoding/kaggle/irrigation/.claude/ralph-x-log.md (append)
- Format: bullet points with source (discussion title)
- Work autonomously. Do NOT ask questions.
PROMPTEOF
)" --max-turns 30

echo "━━━ Discussion Research Complete ━━━"

# ━━━ Main Loop ━━━
for i in $(seq 1 $MAX_ITER); do
  echo "━━━ Ralph-X iteration $i/$MAX_ITER ━━━"

  # Check if all checklist items are done
  if ! grep -q '^\- \[ \]' "$CHECKLIST_FILE" 2>/dev/null; then
    echo "✅ All checklist items complete!"
    break
  fi

  # ━━━ Stage 1: 전략 분석 + 코드 개발 + 실행 ━━━
  echo "--- Stage 1: Strategy + Develop + Run ---"

  claude -p "$(cat << PROMPTEOF
You are in a Ralph-X loop. Iteration $i/$MAX_ITER. Stage: Strategy + Development.

Working directory: /Users/ram/programming/vibecoding/kaggle/irrigation

Task: 새로운 전략으로 irrigation prediction trial을 만들고 실행하라.

Context:
- Competition: playground-series-s6e4 (3-class: Low/Medium/High, metric: accuracy)
- Current best OOF: $(cat "$BEST_SCORE_FILE")
- Data: data/train.csv (630K rows), data/test.csv (270K rows), data/original/irrigation_prediction.csv (10K auxiliary)

Steps:
1. Read $LOG_FILE for discussion insights and previous trial results
2. Read $PROJECT_DIR/TRIALS.md for all tried strategies
3. Propose a NEW strategy that hasn't been tried yet (다른 접근법, FE, 모델, 앙상블 등)
4. Determine next trial number by checking existing submissions/ dirs
5. Create submission dir: submissions/sub_03/trial_NNN_<name>/
6. Write trial code following the pattern in submissions/sub_02/trial_002_fe_catboost/trial_002_fe_catboost.py
7. Run the trial: python submissions/sub_03/trial_NNN_<name>/trial_NNN_<name>.py
8. Ensure it saves: results.json, oof_preds.npy, test_preds.npy, submission.csv

Code requirements:
- DATA_DIR = Path(__file__).resolve().parents[3] / "data"
- OUT_DIR = Path(__file__).resolve().parent
- 5-fold StratifiedKFold CV
- Save OOF accuracy in results.json as {"oof_accuracy": float, "strategy": str}
- submission.csv with columns [id, Irrigation_Need] (string labels: Low/Medium/High)

IMPORTANT:
- Work autonomously. Do NOT ask questions.
- Append trial summary to $LOG_FILE
- Do NOT modify TRIALS.md (Stage 2 handles that)
PROMPTEOF
)" --max-turns 50

  # ━━━ Stage 2: 검증 + 제출 판단 ━━━
  echo "--- Stage 2: Verify + Submit Decision ---"

  claude -p "$(cat << PROMPTEOF
You are in a Ralph-X loop. Iteration $i/$MAX_ITER. Stage: Verify + Submit.

Working directory: /Users/ram/programming/vibecoding/kaggle/irrigation

Task: 최신 trial 결과를 검증하고, TRIALS.md를 업데이트하라.

Steps:
1. Find the latest trial in submissions/sub_03/ (most recent by number)
2. Read its results.json to get OOF accuracy
3. Update TRIALS.md: add a new row with trial info
4. Compare OOF score with current best: $(cat "$BEST_SCORE_FILE")
5. If new OOF > current best:
   - Update $BEST_SCORE_FILE with new score
   - echo "LOCAL_BEST_UPDATED" to $PROJECT_DIR/.claude/ralph-x-status.txt
6. If new OOF <= current best:
   - echo "NO_IMPROVEMENT" to $PROJECT_DIR/.claude/ralph-x-status.txt
7. Count total trials in TRIALS.md. If >= 12, mark checklist item done:
   - Edit $CHECKLIST_FILE: change "- [ ] 12개 이상 전략 시도" to "- [x] 12개 이상 전략 시도"
8. Append verification summary to $LOG_FILE

IMPORTANT:
- Work autonomously. Do NOT ask questions.
- Be accurate with scores. Read results.json directly.
PROMPTEOF
)" --max-turns 20

  TRIALS_DONE=$((TRIALS_DONE + 1))

  # ━━━ 3iter마다 제출 판단 ━━━
  if [ $((i % 3)) -eq 0 ]; then
    echo "--- Submit Check (every 3 iterations) ---"
    STATUS=$(cat "$PROJECT_DIR/.claude/ralph-x-status.txt" 2>/dev/null || echo "UNKNOWN")

    # Check if ANY of the last 3 iterations had a local best update
    if [ "$STATUS" = "LOCAL_BEST_UPDATED" ]; then
      echo "🚀 Local best updated! Submitting to Kaggle..."

      # Find the best trial's submission.csv
      claude -p "$(cat << PROMPTEOF
You are in a Ralph-X loop. Stage: Auto-submit.

Working directory: /Users/ram/programming/vibecoding/kaggle/irrigation

Task: 가장 좋은 OOF score를 가진 trial의 submission.csv를 Kaggle에 제출하라.

Steps:
1. Read TRIALS.md to find the trial with the highest Val Score
2. Locate its submission.csv file
3. Submit using: kaggle competitions submit -c playground-series-s6e4 -f <path_to_submission.csv> -m "<trial_name>: <strategy_summary>, OOF acc <score>"
4. Wait 10 seconds, then check result: kaggle competitions submissions -c playground-series-s6e4
5. Update SUBMISSIONS.md with the new submission result
6. Append submission result to $LOG_FILE

IMPORTANT:
- Work autonomously. Do NOT ask questions.
- Include OOF accuracy and strategy in the submission message
PROMPTEOF
)" --max-turns 20

      echo "✅ Submission complete"
    else
      echo "⏭️ No local best update in recent iterations, skipping submit"
    fi
  fi

  echo "━━━ Iteration $i complete (${TRIALS_DONE} trials done) ━━━"
done

echo "🏁 Ralph-X finished after $i iterations, $TRIALS_DONE trials completed"
