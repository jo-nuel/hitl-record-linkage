# Evaluation Summary

The final comparison uses exactly three methods: Human-only Clerical Review Baseline, AI-only EMPI Matcher, and AI + HITL Grey-Zone Review.

AI-only treats grey-zone pairs as unresolved non-positive predictions, so true links in the grey zone count as missed links.

```
                             Method  Precision  Recall  F1-score  False positives  False negatives  Candidate pairs reviewed  Review workload percentage Estimated review time                                                            Key interpretation
Human-only Clerical Review Baseline        1.0   0.998     0.999                0                8                    151017                      100.00     2,265,255 seconds            Highest workload because every blocked candidate pair is reviewed.
               AI-only EMPI Matcher        1.0   0.806     0.892                0              972                         0                        0.00             0 seconds No manual workload, but grey-zone true links are missed when left unresolved.
         AI + HITL Grey-Zone Review        1.0   0.994     0.997                0               32                       952                        0.63        14,280 seconds         Best accuracy-efficiency trade-off by reviewing only grey-zone pairs.
```
