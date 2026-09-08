"""Lab 3 starter: extractive QA post-processing."""

import numpy as np


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20
):


    # نجيب أفضل احتمالات للبداية والنهاية بدل تجربة كل الـtokens
    starts = np.argsort(start_logits)[-top_k:]
    ends = np.argsort(end_logits)[-top_k:]

    best_score = float("-inf")
    best_start = None
    best_end = None

    # نجرب أفضل start مع أفضل end
    for s in starts:
        for e in ends:

            # token 0 غالبًا [CLS] ويستخدم لتمثيل no-answer
            if s == 0 or e == 0:
                continue

            # نرفض span مقلوب: النهاية ما تكون قبل البداية
            if e < s:
                continue

            # نرفض الإجابة إذا كانت أطول من الحد المسموح
            if e - s + 1 > max_answer_len:
                continue

            # بعض الـtokens مثل special/question tokens ما لها offset في الـcontext
            if offsets[s] is None or offsets[e] is None:
                continue

            # درجة الـspan = درجة البداية + درجة النهاية
            score = float(start_logits[s] + end_logits[e])

            # نحفظ أفضل span
            if score > best_score:
                best_score = score
                best_start = s
                best_end = e

    # إذا ما لقينا أي span صالح
    if best_start is None:
        return {
            "answer": None,
            "reason": "no_valid_span",
        }

    # إذا no-answer أقوى من أفضل إجابة بأكثر من threshold
    # نرفض الإجابة بدل ما نخترع answer
    if null_score - best_score > null_threshold:
        return {
            "answer": None,
            "reason": "no_answer_in_context",
            "margin": float(null_score - best_score),
        }

    # نحول token positions إلى character offsets داخل النص
    start_char = offsets[best_start][0]
    end_char = offsets[best_end][1]

    return {
        "answer": (start_char, end_char),
        "start": start_char,
        "end": end_char,
        "score": best_score,
    }