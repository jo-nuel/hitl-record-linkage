# Evaluation Summary

The central final evaluation for the project is written by the active-learning experiment to `outputs/tables/final_research_evaluation.csv`. It compares Human-only Clerical Review Baseline, AI-only ML Matcher, AI + HITL Active Learning Matcher, Random Sampling HITL Baseline, and Hybrid EMPI Baseline.

The table below is supporting evidence for the operational EMPI grey-zone workflow. It uses exactly three methods: Human-only Clerical Review Baseline, AI-only EMPI Matcher, and AI + HITL Grey-Zone Review.

AI-only treats grey-zone pairs as unresolved non-positive predictions, so true links in the grey zone count as missed links.

Formal benchmark metrics are generated from the evaluation pipeline. The AI + HITL result uses simulated grey-zone review based on FEBRL ground truth to represent an idealised human reviewer. Live reviewer decisions in Streamlit are stored for demonstration and audit logging, but they do not automatically overwrite formal benchmark metrics unless the pipeline is explicitly rerun in merge mode.

```
                             Method  Precision  Recall  F1-score  False positives  False negatives  Candidate pairs reviewed  Review workload percentage Estimated review time                                                            Key interpretation
Human-only Clerical Review Baseline        1.0   0.998     0.999                0                8                    151017                      100.00     2,265,255 seconds            Highest workload because every blocked candidate pair is reviewed.
               AI-only EMPI Matcher        1.0   0.806     0.892                0              972                         0                        0.00             0 seconds No manual workload, but grey-zone true links are missed when left unresolved.
         AI + HITL Grey-Zone Review        1.0   0.994     0.997                0               32                       952                        0.63        14,280 seconds         Best accuracy-efficiency trade-off by reviewing only grey-zone pairs.
```
