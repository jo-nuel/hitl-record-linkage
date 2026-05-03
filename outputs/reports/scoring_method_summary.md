# Scoring Method Summary

The matcher uses a blended EMPI-inspired score. It first attempts to estimate pair-level probability with `recordlinkage.ECMClassifier`. It also calculates a transparent Hybrid EMPI-style evidence score from field-level agreement values. The final model score blends ECM probability with the hybrid score using the configured ECM weight.

## Field Weights

- given_name_sim: 0.16
- surname_sim: 0.20
- date_of_birth_exact: 0.20
- address_sim: 0.16
- suburb_sim: 0.08
- state_exact: 0.05
- postcode_exact: 0.10
- sex_exact: 0.05

Date of birth, surname, postcode, and address receive stronger weight because they provide stronger identity evidence than broad or sometimes missing fields. The hybrid score also applies penalties for strong disagreement in date of birth, surname, postcode, and sex/gender where those fields are available.

Default lower threshold: 0.50

Default upper threshold: 0.80

Pairs at or above the upper threshold become Auto Match. Pairs at or below the lower threshold become Auto Non-match. Pairs between the thresholds are grey-zone cases sent to human review. This makes the method explainable because reviewers can inspect both the total score and the field-level evidence.
