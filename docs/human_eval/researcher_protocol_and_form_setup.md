# Researcher Protocol: Running the Blind Human Study (~15 Subjects)

This guide explains how **Charles, Antoine, and Max** run the human evaluation end to end—from collecting watch histories to packaging blinded MP4s, setting up the Google Form, and analyzing unblinded results.

---

## End-to-End Timeline

1. **Invite & Collect History (~2 min for participant):**  
   Send [`participant_history_guide_en.md`](participant_history_guide_en.md) (or [`_fr.md`](participant_history_guide_fr.md)) to each subject. They visit `https://charlesdupouy.github.io/nlp-multiagent-storyboarding/` and send back `youtube_shorts_history.json`.

2. **Ingest & Split into 3 Disjoint Parts (`3 × 30` Shorts):**
   ```bash
   python3 src/tools/import_user_history.py --input /path/to/youtube_shorts_history.json --user alice --split 3 --cap 90
   ```
   This creates `alice_part1.json`, `alice_part2.json`, and `alice_part3.json` in `data/user_histories/`.

3. **Generate All 6 Storyboards & Videos (`3 Baseline + 3 Agentic`):**
   ```bash
   for part in part1 part2 part3; do
     python3 main.py --history data/user_histories/alice_${part}.json --baseline
     python3 main.py --history data/user_histories/alice_${part}.json
     python3 storyboard_to_video.py outputs/alice_${part}/baseline/baseline_storyboard_alice_${part}_*.json
     python3 storyboard_to_video.py outputs/alice_${part}/agentic/agentic_storyboard_alice_${part}_*.json
   done
   ```
   *(Reminder: Unsplash demo apps allow 50 requests/hour ≈ 3–4 full 6-video participant batches per hour, or set `PEXELS_API_KEY` in `.env` for seamless spillover.)*

---

## Step 4: Build the Google Form Once (~5 Minutes)

Create **one** Google Form divided into **4 sections** (using exact question titles below ensures `--analyze-csv` parses every column automatically):

### Section 1: Participant Info & Brief Framing
- **Form Description:**  
  *"Thanks for taking ~15–20 min to rate 6 short personalized video prototypes (3 pairs). Sound ON! Remember: these are animated storyboard pitches (stock photos + voiceover + bottom shot description), not finished YouTube Shorts—judge the idea, storytelling, fit to your tastes, and overall appeal."*
- **Question 1 (Short answer, Required):**  
  `Participant ID` *(Tell each friend their assigned pseudonym, e.g. `alice`, or use a pre-filled Form URL!)*

### Section 2 (`Pair 1`), Section 3 (`Pair 2`), Section 4 (`Pair 3`)
Create Section 2 (`Pair 1`) cleanly, then **Duplicate Section** twice and replace `Pair 1` with `Pair 2` and `Pair 3`. Inside **Section `Pair {k}`** (`k = 1, 2, 3`):

1. `Pair {k} — Video A: Personal Relevance (1–5)`  *(Linear scale `1` = Unrelated / Copy to `5` = Great fresh fit)*
2. `Pair {k} — Video A: Creativity (1–5)`  *(Linear scale `1` = Cliché / Remake to `5` = Very original)*
3. `Pair {k} — Video A: Engagement (1–5)`  *(Linear scale `1` = Boring / Confusing to `5` = Hooked end-to-end)*
4. `Pair {k} — Video A: Overall Preference (1–5)`  *(Linear scale `1` = Instant skip to `5` = Would love to watch)*
5. `Pair {k} — Video B: Personal Relevance (1–5)`  *(Linear scale `1`–`5`)*
6. `Pair {k} — Video B: Creativity (1–5)`  *(Linear scale `1`–`5`)*
7. `Pair {k} — Video B: Engagement (1–5)`  *(Linear scale `1`–`5`)*
8. `Pair {k} — Video B: Overall Preference (1–5)`  *(Linear scale `1`–`5`)*
9. `Pair {k} — Which video would you rather watch?`  *(Multiple choice, Required)*  
   - `Video A`
   - `Video B`
   - `No preference / Tie`
10. `Pair {k} — Why? (Optional)`  *(Short answer, Optional)*

> **Tip (Pre-filled Participant ID):** In Google Forms click `⋮ → Get pre-filled link`, type `TEST_USER` in `Participant ID`, copy the link, and look for `entry.XXXXXXXXX=TEST_USER`. Passing `--participant-entry entry.XXXXXXXXX` to our packager stamps every friend's folder with a 1-click link that already fills in their ID!

---

## Step 5: Package Blinded Folders & Send to Participants

Run [`prepare_human_eval.py`](../../evaluation/prepare_human_eval.py):

```bash
python3 evaluation/prepare_human_eval.py \
  --form-url "https://docs.google.com/forms/d/e/YOUR_FORM_ID/viewform" \
  --participant-entry "entry.XXXXXXXXX" \
  --zip
```

What this writes inside `outputs/human_eval_packages/`:
- `outputs/human_eval_packages/<participant>/` *(and `<participant>.zip`)*:
  - `Pair1_VideoA.mp4` ... `Pair3_VideoB.mp4` — deterministically balanced `AB`/`BA` randomization per pair, zero system tags in filenames or container metadata!
  - `INSTRUCTIONS_EN_FR.md` — contains their exact Participant ID and personalized Form URL.
- **Keep private (DO NOT SHARE):** `outputs/human_eval_packages/blinding_key.csv` & `blinding_key.json`.

Send each participant **only their own folder / `.zip`** (via Google Drive / WhatsApp / Email) + [`participant_grading_guide_en.md`](participant_grading_guide_en.md) or [`participant_grading_guide_fr.md`](participant_grading_guide_fr.md).

---

## Step 6: Unblind & Analyze Results

Once participants submit the Google Form:
1. In Google Forms → **Responses** → **Link to Sheets** → **File → Download → Comma Separated Values (.csv)**.
2. Run:
   ```bash
   python3 evaluation/prepare_human_eval.py \
     --analyze-csv /path/to/form_responses.csv \
     --key outputs/human_eval_packages/blinding_key.json
   ```
3. Outputs **`human_eval_summary.txt`** (means, deltas, 95% bootstrap CIs, paired Wilcoxon p-values, pairwise win rate, per-user majority) and **`human_eval_unblinded.csv`** cleanly alongside your CSV!
