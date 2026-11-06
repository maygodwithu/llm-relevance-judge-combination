"""APO-CF meta-prompt used by the ESCI proposer (product domain, level "full", no highlighted span; constraint variants shorter / similar / bare).
The template and constraint strings below are copied verbatim from the original APO-CF implementation (A. Choi, APO-CF, Figure 3 prompt with the length constraint axis of this study);
`build_apocf_prompt` has the same signature and output for the arguments used here and rejects the arguments of the other variants of the original implementation."""

APOCF_PROMPT_TEMPLATE_ESCI = """You are an AI model tasked with improving the clarity and effectiveness of an instruction for a binary classification task. The task requires determining whether a candidate product is relevant to a shopper's search query. Your goal is to refine the instruction to minimize classification errors based on the confusion matrix.

## Task Description:
- You are given an 'instruction' that guides the model to classify a candidate product as relevant to a search query ("Relevant") or not ("Irrelevant").
- You are also provided with a 'confusion matrix', which contains information about the model's classification performance.
- Your job is to analyze the confusion matrix, identify potential weaknesses in the instruction, and revise the instruction to reduce the classification errors (FP/FN) shown in the confusion matrix.
- Your revision should ensure that the criteria for responding "Relevant" or "Irrelevant" are clearly defined and aligned with minimizing errors.

## Current Instruction:
"{CURRENT_INSTRUCTION}"

## Confusion Matrix from Training Data:
- True Negatives (TN): {TN}
- False Positives (FP): {FP}
- False Negatives (FN): {FN}
- True Positives (TP): {TP}

### Instruction Refinement Goals:
- If False Positives (FP) are high: The instruction is too lenient, letting products that do not truly match the query be classified as "Relevant." Tighten the specific criterion causing this — but the revision must still classify genuine matches as "Relevant"; do not make the instruction so strict that it starts rejecting reasonable matches over trivial differences.
- If False Negatives (FN) are high: The instruction is too restrictive, preventing genuinely suitable products from being classified as "Relevant." Loosen the specific over-strict interpretation causing this (e.g., minor wording or presentation differences) — but the revision must still classify a genuinely different product type, a substitute, an accessory, or an unrelated item as "Irrelevant"; do not make the instruction broadly lenient (e.g., default to "Relevant" when uncertain), and do not fix this by adding a checklist of attributes to verify instead.
- If both FP and FN are high: The instruction is unclear, leading to inconsistent classification. Improve clarity by defining a precise boundary for what counts as a matching product.
- If neither FP nor FN is significantly high: The instruction is already effective, but slight optimization may improve precision and recall balance.

{CONSTRAINT}

### Output Format:
Respond with only the revised instruction text."""

APOCF_CONSTRAINT_SHORTER = """### Constraint:
Revise the instruction to fix the FP/FN pattern above, and make the result shorter than the current instruction — condense, merge, or remove parts that are not essential rather than adding new material. Do not shorten at the cost of damaging the instruction's meaning or function. The revised instruction MUST explicitly tell the model to respond with only one word, either "Relevant" or "Irrelevant" — do not explain. Never drop or weaken this output-format requirement."""

APOCF_CONSTRAINT_SIMILAR = """### Constraint:
Revise the instruction to fix the FP/FN pattern above, while keeping the result at roughly the same length as the current instruction — adjust or reword existing parts rather than substantially adding or removing content."""

APOCF_CONSTRAINT_BARE = ""


APOCF_CONSTRAINTS = {"shorter": APOCF_CONSTRAINT_SHORTER, "similar": APOCF_CONSTRAINT_SIMILAR, "bare": APOCF_CONSTRAINT_BARE}


def build_apocf_prompt(current_instruction: str, tn: int, fp: int, fn: int, tp: int, highlighted_span: str = None, domain: str = "passage",
                       constraint_variant: str = "similar", level: str = "full", examples_text: str = None) -> str:
    if domain != "product" or level != "full" or highlighted_span or constraint_variant not in APOCF_CONSTRAINTS:
        raise NotImplementedError("only the configuration used in the paper (product domain, full level, no highlighted span, constraint shorter / similar / bare) is included")
    examples_section = f"\n## Recent Misclassification Examples:\n{examples_text}\n" if examples_text else ""
    return APOCF_PROMPT_TEMPLATE_ESCI.format(CURRENT_INSTRUCTION=current_instruction, TN=tn, FP=fp, FN=fn, TP=tp, CONSTRAINT=APOCF_CONSTRAINTS[constraint_variant],
                                             MISCLASSIFIED_EXAMPLES_SECTION=examples_section)
