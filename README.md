# Instamart Customer Intelligence & Support Copilot

## Overview

This is a reproducible, offline capstone implementation for a fictional grocery-delivery business. It joins four workflows in one repository: (A) data quality, SQLite and SQL analytics; (B) customer segmentation and lapse prediction; (C) privacy-aware ticket triage; and (D) policy retrieval plus a tool-using refund agent. All input data is synthetic. All monetary values are INR. All model calls use the supplied local `mock_llm.py`; no API key, hosted model, network call or paid service is used.

The supplied `generate_data.py` and the offline `mock_llm.py` are at the repository root. Generated datasets, analytical outputs, the SQLite database, and run checkpoints are included so the repository can be inspected without running it first.

## Reproduce

Run these commands from a clean clone on Python 3.11+ (the pinned packages below were tested with Python 3.13):

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   ```
   Windows PowerShell: `.venv\\Scripts\\Activate.ps1`; macOS/Linux: `source .venv/bin/activate`.
2. Install the pinned dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Regenerate all raw synthetic CSVs and policy text files:
   ```bash
   python generate_data.py
   ```
   The final line is `generated: customers=1200 products=40 orders_raw=13084 tickets=164`.
4. Build clean CSVs, the SQLite database and SQL analysis:
   ```bash
   python scripts/part_a.py
   ```
5. Build customer segments, lapse models and retention list:
   ```bash
   python scripts/part_b.py
   ```
6. Run ticket redaction, classification, retries, budget demo and routing:
   ```bash
   python scripts/part_c.py
   ```
7. Run policy RAG evaluation, seven agent scenarios, safe-stop demos and crash/resume:
   ```bash
   python scripts/part_d.py
   ```
8. Verify the Part A data/database checkpoints:
   ```bash
   python scripts/validate.py
   ```
   Expected validation output: `ALL CHECKPOINTS PASSED`.

`part_c.py` and `part_d.py` recreate their generated JSON/JSONL/CSV outputs and state files each time. Their timestamps in the refund ledger naturally differ between runs; the outcomes, model behavior, data counts and scoring metrics are deterministic. Retry waits include the required jitter and therefore are not byte-identical between runs.

## Part A

### Raw-data audit

| Table | Shape (rows × columns) | Column | Raw dtype | Missing / blank | Percent |
|---|---:|---|---|---:|---:|
| customers | 1,200 × 5 | customer_id | object | 0 | 0.00% |
| customers | 1,200 × 5 | city | object | 0 | 0.00% |
| customers | 1,200 × 5 | signup_date | object | 0 | 0.00% |
| customers | 1,200 × 5 | age | object | 61 | 5.08% |
| customers | 1,200 × 5 | gender | object | 30 | 2.50% |
| products | 40 × 4 | product_id | object | 0 | 0.00% |
| products | 40 × 4 | product_name | object | 0 | 0.00% |
| products | 40 × 4 | category | object | 0 | 0.00% |
| products | 40 × 4 | price_inr | int64 | 0 | 0.00% |
| orders | 13,084 × 9 | order_id | object | 0 | 0.00% |
| orders | 13,084 × 9 | customer_id | object | 0 | 0.00% |
| orders | 13,084 × 9 | product_id | object | 0 | 0.00% |
| orders | 13,084 × 9 | quantity | int64 | 0 | 0.00% |
| orders | 13,084 × 9 | order_ts | object | 0 | 0.00% |
| orders | 13,084 × 9 | status | object | 0 | 0.00% |
| orders | 13,084 × 9 | discount_pct | int64 | 0 | 0.00% |
| orders | 13,084 × 9 | order_amount | object | 53 | 0.41% |
| orders | 13,084 × 9 | delivery_minutes | object | 2,541 | 19.42% |

Empty strings in object columns are counted as missing for this audit. The raw audit finds 119 exact duplicate order rows, 656 `pending` rows, 12,309 distinct order IDs, 120 non-numeric order amounts, 106 negative delivery durations and 2,541 blank delivery durations. In customers, 61 ages are blank, 6 ages are outside 10–100, there are 24 raw city spellings, 10 distinct nonblank gender spellings and 30 blank genders.

### Cleaning rules and checkpoints

- **Customers:** trim and title-case city; map gender variants to `male`, `female`, or `unknown`; convert ages outside 10–100 to missing; fill all 67 missing/invalid ages (61 blanks + 6 invalid values) with the median age for the customer's city using `groupby('city').transform('median')`; convert `signup_date` to ISO date text. Final city counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Final genders: female 610, male 560, unknown 30. No ages are missing; every age is 18–60.
- **Orders:** parse timestamps; remove 119 exact duplicate rows (13,084 → 12,965); keep the latest timestamp per order ID, which removes 656 earlier `pending` records (12,965 → 12,309); coerce amount to numeric and fill missing values from product price × quantity × (1 − discount / 100), rounded to two decimals; set negative delivery durations to missing; fill missing durations only for completed orders with the completed-order median for the product category. Cancelled orders retain missing delivery duration. The raw export contains 120 nonnumeric amounts, of which 112 survive final-record selection and are imputed; no amount is missing after cleaning. A total of 774 missing/invalid completed-order delivery durations are imputed after final-record selection.
- Final orders: 12,309 unique order IDs; 11,110 completed and 1,199 cancelled; no pending rows; 0 missing amounts; 0 missing delivery durations among completed orders and 1,199 among cancelled orders. Completed revenue is **INR 2,998,668.80**.
- `data/clean/customers.csv`, `products.csv` and `orders.csv` are written to `data/clean/`. `data/instamart.db` contains `customers`, `products` and `orders`; `order_ts` is stored as `YYYY-MM-DD HH:MM:SS`. `scripts/validate.py` checks more than eight data/database checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis

The seven executable queries are saved in `sql/queries.sql` and run against SQLite with `pandas.read_sql_query`.

| Query | Result |
|---|---|
| Q1 — `WHERE` then `HAVING` | 30 customers have completed orders above INR 500 whose qualifying-order total exceeds INR 8,000. |
| Q2 — three-table join | 30 city/category rows; grand total INR 2,998,668.80. Largest cell: Bengaluru / Electronics, INR 233,909.45. |
| Q3 — scalar subquery | 11 products are priced above the average catalogue price. |
| Q4 — `LEFT JOIN` | 25 customers have no completed order. |
| Q5 — CTEs | 550 customers exceed average spend; average completed spend per customer with completed spend is INR 2,552.06. |
| Q6 — ranking windows | 1,175 customers with completed spend ranked by city; `RANK()` differs from `ROW_NUMBER()` for 8 rows. Ties share a rank, while `ROW_NUMBER()` always assigns a unique sequential position. |
| Q7 — running total | 183 daily revenue rows; final cumulative revenue INR 2,998,668.80. |

Top spender in each city from Q6:

| City | Customer | Completed spend |
|---|---|---:|
| Bengaluru | C0082 | INR 12,856.20 |
| Chennai | C0930 | INR 10,151.40 |
| Delhi | C0291 | INR 13,742.60 |
| Hyderabad | C0743 | INR 12,730.05 |
| Mumbai | C0693 | INR 14,766.55 |
| Pune | C0426 | INR 10,377.10 |

### pandas cross-check and EDA

The `pd.pivot_table(..., margins=True)` city/category grand total is INR 2,998,668.80, matching SQL Q2.

| City | Completed amount sum | Mean amount | Order count |
|---|---:|---:|---:|
| Bengaluru | INR 523,643.40 | INR 276.47 | 1,894 |
| Chennai | INR 463,629.15 | INR 272.72 | 1,700 |
| Delhi | INR 502,829.80 | INR 243.62 | 2,064 |
| Hyderabad | INR 528,589.90 | INR 284.95 | 1,855 |
| Mumbai | INR 479,058.50 | INR 297.00 | 1,613 |
| Pune | INR 500,918.05 | INR 252.48 | 1,984 |

Completed order amount: mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, Q3 INR 297.00, IQR INR 232.20, lower fence −INR 283.50 and upper fence INR 645.30. There are no values below the lower fence and 897 outliers above the upper fence. Correlations: amount/discount −0.095, amount/quantity +0.087, amount/delivery time −0.003, discount/quantity approximately −0.000, discount/delivery time −0.010 and quantity/delivery time −0.013. The largest absolute correlation is order amount with discount percentage (−0.095); a correlation this small does not establish causation.

**Interpretation (under 200 words):** This dataset represents synthetic grocery and small-electronics orders across six cities, linked to customers and products. Completed order values are strongly right-skewed: the mean (INR 269.91) is twice the median (INR 135.00), skewness is 4.61, and 897 high-value orders exceed the upper IQR fence. The high outliers are plausible in this synthetic catalogue because premium customers often buy electronics, which cost more than groceries. Data-quality anomalies include duplicated order records, pending snapshots, impossible ages, negative delivery durations and unparseable amounts; the cleaning pipeline removes or imputes them according to documented rules. Amount and discount have the largest absolute pairwise correlation, −0.095, followed by amount and quantity at +0.087. Both associations are weak, and correlation of this size does not establish causation.

## Part B

### RFM and segments

RFM uses completed orders as of `2025-10-01 00:00:00`. The table has 1,175 customers; recency median 19 days, minimum 0 and maximum 180; frequency mean 9.46 and maximum 33; monetary mean INR 2,552.06 and maximum INR 14,766.55.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected because it has the highest silhouette among the business-choice candidates (K=3, 4, 5), a visible inertia drop from K=3 to K=4, and useful customer profiles. K=8 always has lower inertia than K=2 because adding clusters can only keep or reduce within-cluster squared distances; that monotonic decrease rewards more clusters and does not by itself indicate business usefulness or generalisation.

| Segment ID | Business name | Customers | Mean recency (days) | Mean frequency | Mean monetary | Premium? | Recommended action |
|---:|---|---:|---:|---:|---:|---|---|
| 0 | Occasional Essentials | 436 | 28.86 | 3.77 | INR 1,057.62 | No | Send replenishment reminders and low-friction grocery bundles. |
| 1 | Premium | 138 | 24.12 | 6.77 | INR 7,129.70 | Yes | Prioritise personalised service, early access and high-value retention offers. |
| 2 | Frequent Essentials | 419 | 8.77 | 19.40 | INR 3,410.02 | No | Use subscription/reorder nudges and relevant basket add-ons. |
| 3 | Dormant / Lapsed | 182 | 109.04 | 2.20 | INR 685.99 | No | Run a measured win-back campaign and suppress repeated offers after nonresponse. |

The Premium flag belongs to the segment with the highest mean monetary value. Cluster IDs are arbitrary labels, not an ordinal scale. PCA on the scaled RFM matrix explains 0.6343 and 0.2180 of variance in its first two components (total 0.8523). `outputs/customer_segments.csv` has the required columns.

### Lapse definition, split and leakage prevention

The cohort contains 1,156 customers with a completed order before `2025-09-01`; 443 lapsed (38.32%) and 713 did not. The stratified split is 924 training rows and 232 test rows. Predictors use only pre-cutoff completed orders plus the clean customer table: recency, frequency, monetary, average discount, cancellation rate, average delivery time, favourite category, city, age, gender and tenure. No order on or after 2025-09-01 enters the features, and the Part B segment labels are not model features. All encoders and scalers are fitted inside a training-only pipeline; no encoder, scaler or resampler sees test rows. No feature column has a missing value after the documented imputation.

The full-cohort frequency mean is 8.0285; the training-fitted scaler's learned frequency mean is 8.1147. They differ because the scaler is fitted only on the training split, not the full cohort.

### Model tuning and evaluation

Baseline that always predicts “not lapsed”: accuracy 0.6164; lapsed precision 0, recall 0 and F1 0.

| Model | Grid | Best parameters | Best CV F1 | Test confusion matrix `[[TN, FP], [FN, TP]]` | Test precision | Test recall | Test F1 |
|---|---|---|---:|---|---:|---:|---:|
| Logistic regression | `C ∈ {0.01, 0.1, 1, 10, 100}`; `max_iter=1000`; 5-fold CV | `C=100` | 0.6506 | `[[111, 32], [31, 58]]` | 0.6444 | 0.6517 | 0.6480 |
| Random forest | `max_depth ∈ {3, 5, 8}`, `min_samples_leaf ∈ {5, 10, 20}`; 100 trees; 5-fold CV (45 fits) | `max_depth=8`, `min_samples_leaf=5` | 0.6923 | `[[105, 38], [34, 55]]` | 0.5914 | 0.6180 | 0.6044 |

The selected persisted model is logistic regression because its untouched-test F1 is higher (0.6480 vs. 0.6044) and it also has better precision and recall on this split. Missing a lapsing customer loses a chance to retain future orders; wrongly targeting a loyal customer wastes incentives and can train customers to wait for discounts. The chosen threshold/model balances both costs on this test set, while the business should adjust it based on campaign budget and observed uplift.

### Class-balance experiment

Training-only random oversampling duplicates lapsed rows with replacement until the training classes are balanced at 570 each. Evaluation remains on the original, unchanged test set.

| Logistic model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original training distribution | 0.6444 | 0.6517 | 0.6480 |
| Oversampled training set | 0.5891 | 0.8539 | 0.6972 |

Recall moves most, rising by about 0.20; precision decreases slightly. The retention team catches substantially more at-risk customers but should expect more false-positive outreach.

### Interpretation and persistence

Top five random-forest importances: `frequency` 0.26896, `monetary` 0.23986, `recency` 0.11455, `avg_delivery` 0.06681 and `cancellation_rate` 0.05643. Frequency and monetary history dominate, with recency next. Three largest-magnitude logistic coefficients are: frequency −1.69733 (higher frequency reduces predicted lapse odds), favourite category Electronics −1.13517 (relative to the dropped reference category, associated with lower lapse odds), and favourite category Grocery −0.44930 (also lower lapse odds relative to the reference). These are associations learned from synthetic data, not causal effects.

The selected pipeline is saved to `outputs/lapse_model.joblib`, reloaded with `joblib.load`, and produces exactly identical test predictions. `outputs/retention_list.csv` contains the top 10 Premium-segment customers from the test set ranked by predicted lapse probability; 3 of these 10 actually lapsed. **This retention list is illustrative and not a leakage-free evaluation**, because the Premium segment label is derived from whole-window RFM that includes September, even though the lapse model features themselves obey the cutoff.

## Part C

### Guardrails, redaction and resilience

Ticket input is read with `keep_default_na=False`. Empty/whitespace-only tickets are `skipped_empty` (4); inputs above 400 estimated tokens are `flagged_long` (4; each is about 734 tokens). Neither group is sent to the model. Redaction is applied in the required order: email → `[EMAIL]`, phone → `[PHONE]`, order ID → `[ORDER_ID]`. Substitutions across all 164 tickets: 55 emails, 91 phone numbers, 132 order IDs, 278 total. Rechecking the redacted text with all three patterns finds 0 matches. The model's user-message content is exactly the redacted ticket text, without any wrapper or other content.

Both prompts contain `CLASSIFY`, the four allowed categories and a JSON-only response contract. The few-shot prompt includes an `EXAMPLES:` line and labelled examples including fraud. Every call is retried for transient failures up to three retries with exponential backoff and jitter: `0.05 × 2**n × uniform(0.9, 1.15)`. Each complete run recovered all 19 simulated `TransientLLMError`s; the retry log records ticket, retry number and wait.

All replies are validated. Valid JSON with one of the allowed categories is `classified_json`; non-JSON output is accepted as `classified_keyword` only when the required lowercase-regex extraction yields exactly one allowed category; invalid JSON/category is `flagged_invalid_output` and sent to human review. Per full run: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 `skipped_empty` and 4 `flagged_long`.

### Zero-shot versus few-shot

| Run | Classified | Correct | Accuracy | Total tokens | Tokens / classified ticket |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 147 | 126 | 85.71% | 11,740 | 79.86 |
| Few-shot | 147 | 147 | 100.00% | 24,220 | 164.76 |

Few-shot costs 12,480 more tokens overall, about 84.90 additional tokens per classified ticket, but fixes all 21 classification errors in this synthetic run. This extra cost is justified for high-risk or ambiguous tickets (especially fraud) where routing mistakes are expensive; a zero-shot route can remain a cheaper first pass only if invalid or high-risk cases are sent to human review.

Zero-shot confusion table (rows are true category; columns are predicted category; only classified tickets are included):

| True \ Predicted | billing | delivery | technical | fraud |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

### Budget and final routing

The zero-shot budget demonstration uses `TOKEN_BUDGET=6000`; it ends with 76 tickets marked `budget_exhausted` after the running total crosses the threshold. The last successful response takes the total to 6,031 tokens; subsequent calls are blocked when the running total is already at least 6,000. `outputs/triage_budget_demo.csv` records this run.

The final `outputs/triage_results.csv` is generated by a few-shot run with `TOKEN_BUDGET=100000` and contains 164 rows and exactly these columns: `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`. `skipped_empty`, `flagged_long`, `flagged_invalid_output` and `budget_exhausted` go to `human_review` at NORMAL priority; fraud goes to `security_team` at HIGH priority; otherwise Premium customers go to `premium_desk` at HIGH priority; all others go to `support_queue` at NORMAL priority. Unknown segment IDs are treated as non-premium. Categories and segment names are blank when unavailable. Rule-validated JSON/keyword classifications are routed automatically under these rules; invalid model output, empty/long input and budget exhaustion are explicitly routed to humans.

## Part D

### Policy RAG

Five policy documents are split into 40-word chunks with 10-word overlap: refund 4, delivery 3, escalation 3, payment 3, privacy 2 (15 chunks total). A `TfidfVectorizer` is fit on chunk text and `cosine_similarity` ranks query results; the top three results for `refund policy late delivery` all come from `refund_policy.txt`. TF (term frequency) measures how often a term occurs in a chunk, while IDF downweights terms that appear across many chunks. TF-IDF can miss semantically equivalent phrasings that use different words because it relies on lexical overlap.

The `answer(question)` function retrieves three chunks, prefixes each context line with its source filename, then sends exactly two messages to `mock_llm.chat`: a `POLICY ASSISTANT` system instruction and the required `CONTEXT:` / `QUESTION:` user content. It returns the answer and source list.

| Evaluation | Answerable doc in top 3 | Expected fact present | Unanswerable questions correctly fall back |
|---|---:|---:|---:|
| 40 words, 10-word overlap | 12/12 | 12/12 | 2/2 |
| 40 words, no overlap | 12/12 | 10/12 | 2/2 |

With no overlap, the full-refund delivery-time question and the payment-failure return-time question fail the expected-fact check: the answer facts sit across chunk boundaries, so no single retrieved chunk reliably contains the needed sentence. The fallback for the two unsupported questions begins `I could not find`.

### Tools and refund agent

`TOOLS_SPEC` documents the five tools and JSON-style parameter schemas; `TOOL_MAP` maps their names to Python functions. `get_order` queries `data/instamart.db`; `search_policy` reuses the same retriever; `calculate_refund` returns the full order amount only for completed orders delivered in more than 45 minutes, otherwise INR 0; `issue_refund` and `escalate_to_human` append JSON lines with timestamps and tool names to `outputs/refund_ledger.jsonl`. Refunds above INR 1,000 are escalated rather than issued automatically. All seven simulated requests are treated as raised within the seven-day policy window.

| Scenario | Status | Refund amount | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | INR 128.25 | 5 | 992 |
| S2 | REFUND | INR 135.00 | 5 | 992 |
| S3 | ESCALATE | INR 1,999.00 | 5 | 979 |
| S4 | DENIED | INR 0.00 | 4 | 681 |
| S5 | DENIED | INR 0.00 | 3 | 394 |
| S6 | NOT_FOUND | INR 0.00 | 2 | 110 |
| S7 | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 | 3,114 |

The scenario outcomes and refund amounts match `agent_scenarios.csv` (within INR 0.01). S1 requires 5 model calls. S7 uses all six model calls and safely falls back with `FALLBACK_MAX_ITERATIONS`.

**Stop-safely demonstrations:** S1 with `token_limit=200` ends as `FALLBACK_TOKEN_LIMIT`; S1 with `time_limit_seconds=0` ends as `FALLBACK_TIMEOUT` before its first model call. Each fallback includes a human-handoff summary and the completed steps.

**Crash/resume demonstration:** S1 is crashed immediately after the second tool execution, after its checkpoint is written. `resume_agent('S1_crash_resume')` resumes from the saved message list and executes only the remaining tools. Tool execution order across both parts is exactly `get_order` → `search_policy` → `calculate_refund` → `issue_refund`; there are four executions, none repeated. Final result: `REFUND`, INR 128.25. Checkpoints in `state/` include the run ID, scenario ID, full messages, token count, step number, tool history and steps.

**ReAct versus plan-and-execute (under 250 words):** ReAct chooses one action from the latest observation, so the S1 happy path uses five model calls: get order, retrieve policy, calculate, issue, and final response. It uses more calls and tokens than a plan-and-execute design that asks for a complete plan once, then replans only on error. However, a fixed plan is fragile when an observation differs from expectation: `NOT_FOUND`, a cancelled order, a delivery under 45 minutes, or a refund above INR 1,000 must change the next action. ReAct can inspect each tool result and stop or escalate based on that result, while a plan needs explicit validation and a replanning path to achieve equivalent safety. For a money-moving workflow, I would keep the guarded ReAct pattern because it prioritises observation-based decisions and auditable tool actions, while enforcing iteration, token and wall-clock limits. A plan-and-execute variant could reduce routine-call overhead, but should never execute a precomputed refund action without rechecking the order status, delivery duration and escalation threshold.

## Limitations and Next Steps

- All data and policies are synthetic, and `mock_llm.py` is deterministic; these scores do not establish production accuracy, policy coverage or business impact.
- The selected lapse model reaches test F1 0.648 on this split, but deployment would require temporal validation, calibration, monitoring, fairness checks, and an experiment measuring actual retention uplift.
- **The Premium flag is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation.** The model's own features use only pre-2025-09-01 orders; the whole-window segment is used only to filter the requested demonstration list.
- Automated triage only routes validated categories. Invalid responses, long/empty inputs and exhausted budgets go to human review; real deployments should also require human confirmation for uncertain or high-impact cases.
- The refund agent uses an append-only simulation ledger and a synthetic SQLite database. Production money movement would require authentication, idempotency keys, transactional state, explicit approval, audit retention and robust duplicate-action protection.
- TF-IDF is lexical rather than semantic. A production retrieval system should add retrieval-quality monitoring and a larger labelled evaluation set before changing the retrieval method.
