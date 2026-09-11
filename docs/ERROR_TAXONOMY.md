# Error Taxonomy Starter

1. Label ambiguity
2. Arabic orthographic variation
3. Dialect or code-switching
4. Entity boundary or clitic alignment
5. Long-context truncation
6. Retrieval relevance mismatch
7. Preprocessing or serving skew
8. Annotation defect

9. Topic confusion: park paths / park irrigation / playground maintenance.

## Lab 6 Step 4

[Analysis, histogram and prioritised fixes](ERROR_REVIEW_ANALYSIS.md).

[120 case annotations](../data/eval/error_review_120.csv) contain assistant-authored draft categories and evidence. Human reviewer fields remain blank.

Secondary tags describe observed text features only: street (road/street in place name), orth (orthographic variation), elong (elongated polite request), emoji, mask (PII placeholder), space (outer whitespace). These do not establish causal failure mechanisms.

Each human reviewer should read the text and assess the draft. Record both names and set review_status to reviewed only after their agreement. Categories 1-8 remain available if supported by evidence.
