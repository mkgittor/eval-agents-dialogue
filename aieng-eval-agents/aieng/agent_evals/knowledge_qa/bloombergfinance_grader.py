"""Bloomberg Financial News grader for evaluating AI-generated financial analysis.

This module provides two evaluation approaches:

1. **Quality grader** (``evaluate_bloomberg_async``) — LLM-as-judge scoring
   accuracy, relevance, insight, and clarity without a reference answer.
2. **Ground-truth grader** (``evaluate_bloomberg_groundtruth_async``) — compares
   the agent's answer against a known correct answer using precision, recall, and
   F1 following the DeepSearchQA methodology.
"""

import logging
from enum import Enum
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import Evaluation
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class NewsQuality(str, Enum):
    """Representating Quality of a news."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class BloombergNewsResult(BaseModel):
    """Evaluation result for financial news responses tailored to capital markets."""

    factual_accuracy: float = Field(0.0, description="Are financial figures, dates, and entity names correct? (0-1)")
    financial_completeness: float = Field(0.0, description="Does the response include key metrics: EPS, revenue, YoY change, beat/miss estimates, dividend? (0-1)")
    market_impact: float = Field(0.0, description="Does it explain implications for investors, stock price, or the sector? (0-1)")
    source_quality: float = Field(0.0, description="Does it cite credible sources with proper attribution? (0-1)")
    actionability: float = Field(0.0, description="Could a trader or analyst act on this information? (0-1)")
    hallucination: float = Field(0.0, description="Does the response avoid fabricated or unverifiable claims not supported by sources? (0-1, higher is better)")
    coherence: float = Field(0.0, description="Is the response logically structured, internally consistent, and easy to follow? (0-1)")
    coverage: float = Field(0.0, description="Does the response address all aspects of the question without omitting key topics? (0-1)")

    overall_score: float = Field(0.0, description="Weighted average score (0-1)")
    quality: NewsQuality = Field(default=NewsQuality.POOR)

    explanation: str = Field(default="", description="Grader explanation")

    def to_evaluations(self) -> list[Evaluation]:
        comment = (
            f"Factual Accuracy: {self.factual_accuracy:.2f}\n"
            f"Financial Completeness: {self.financial_completeness:.2f}\n"
            f"Market Impact: {self.market_impact:.2f}\n"
            f"Source Quality: {self.source_quality:.2f}\n"
            f"Actionability: {self.actionability:.2f}\n"
            f"Hallucination: {self.hallucination:.2f}\n"
            f"Coherence: {self.coherence:.2f}\n"
            f"Coverage: {self.coverage:.2f}\n"
            f"Overall: {self.overall_score:.2f}\n\n"
            f"Explanation: {self.explanation}"
        )

        return [
            Evaluation(name="Quality", value=self.quality.value, comment=self.explanation),
            Evaluation(name="Overall", value=self.overall_score, comment=comment),
            Evaluation(name="Factual Accuracy", value=self.factual_accuracy, comment=comment),
            Evaluation(name="Financial Completeness", value=self.financial_completeness, comment=comment),
            Evaluation(name="Market Impact", value=self.market_impact, comment=comment),
            Evaluation(name="Source Quality", value=self.source_quality, comment=comment),
            Evaluation(name="Actionability", value=self.actionability, comment=comment),
            Evaluation(name="Hallucination", value=self.hallucination, comment=comment),
            Evaluation(name="Coherence", value=self.coherence, comment=comment),
            Evaluation(name="Coverage", value=self.coverage, comment=comment),
        ]

    @staticmethod
    def error_evaluations(error_msg: str) -> list[Evaluation]:
        comment = f"Evaluation error: {error_msg}"
        return [
            Evaluation(name="Quality", value="poor", comment=comment),
            Evaluation(name="Overall", value=0.0, comment=comment),
            Evaluation(name="Factual Accuracy", value=0.0, comment=comment),
            Evaluation(name="Financial Completeness", value=0.0, comment=comment),
            Evaluation(name="Market Impact", value=0.0, comment=comment),
            Evaluation(name="Source Quality", value=0.0, comment=comment),
            Evaluation(name="Actionability", value=0.0, comment=comment),
            Evaluation(name="Hallucination", value=0.0, comment=comment),
            Evaluation(name="Coherence", value=0.0, comment=comment),
            Evaluation(name="Coverage", value=0.0, comment=comment),
        ]


class BloombergGraderResponse(BaseModel):
    """Structured grader response."""

    evaluation: dict[str, Any] = Field(
        alias="Evaluation",
        description="Contains scores and explanation"
    )


BLOOMBERG_GRADER_PROMPT = """\
You are a senior capital markets analyst evaluating an AI research assistant's
output about Canadian Big Five bank news. You need this information to make
portfolio and trading decisions.

Evaluate the response on these eight dimensions, each scored 0 to 1:

1. Factual Accuracy (0-1)
- Are financial figures (net income, EPS, revenue, percentages) correct?
- Are dates, entity names, and event descriptions accurate?
- Are there any fabricated or hallucinated claims?

2. Financial Completeness (0-1)
- For earnings questions: does it include net income, EPS, YoY change,
  beat/miss vs estimates, revenue, and dividend changes where applicable?
- For non-earnings questions: does it cover all key facts a capital markets
  analyst would need?
- Score 0.0 if only a single headline fact is given with no supporting detail.
  Score 1.0 if all material financial metrics are present.

3. Market Impact (0-1)
- Does it explain what the news means for investors, stock prices, or the sector?
- Does it provide forward-looking context (analyst outlook, rate expectations,
  peer comparisons)?
- Score 0.0 if it just states facts with no analysis. Score 1.0 if a trader
  could make a decision based on this response.

4. Source Quality (0-1)
- Does it cite specific, credible sources (Bloomberg, Reuters, bank press
  releases, regulatory filings, earnings reports)?
- Are citations formatted with publication name, date, and title/URL?
- Score 0.0 if no sources cited. Score 0.5 if sources mentioned but vague.
  Score 1.0 if properly attributed credible sources.

5. Actionability (0-1)
- Could a portfolio manager or trader act on this information?
- Is the information specific enough (not vague generalizations)?
- Does it distinguish between confirmed facts and speculation?

6. Hallucination (0-1)
- Does the response avoid fabricated, unverifiable, or invented claims not
  supported by cited sources or well-established facts?
- Score 0.0 if significant hallucinations are present. Score 1.0 if every
  claim is grounded and verifiable.

7. Coherence (0-1)
- Is the response logically structured, internally consistent, and easy to follow?
- Does it flow clearly from one point to the next without contradictions?
- Score 0.0 if the response is disjointed or contradictory. Score 1.0 if it
  reads as a well-organized, clear analytical narrative.

8. Coverage (0-1)
- Does the response address all aspects of the question without omitting key topics?
- For multi-part questions, does it handle each part adequately?
- Score 0.0 if major aspects of the question are ignored. Score 1.0 if the
  response fully addresses the scope of the question.

Overall Score: weighted average with these weights:
- Factual Accuracy: 30%
- Financial Completeness: 25%
- Market Impact: 20%
- Source Quality: 15%
- Actionability: 10%

Quality Label based on Overall:
- excellent (>=0.85)
- good (>=0.7)
- fair (>=0.5)
- poor (<0.5)

Return JSON format:

{{
  "Evaluation": {{
    "Factual Accuracy": float,
    "Financial Completeness": float,
    "Market Impact": float,
    "Source Quality": float,
    "Actionability": float,
    "Hallucination": float,
    "Coherence": float,
    "Coverage": float,
    "Overall": float,
    "Quality": "excellent|good|fair|poor",
    "Explanation": "..."
  }}
}}

User Prompt:
{prompt}

AI Response:
{response}
"""


def _parse_bloomberg_result(grader_result: dict[str, Any]) -> BloombergNewsResult:
    factual_accuracy = grader_result.get("Factual Accuracy", 0.0)
    financial_completeness = grader_result.get("Financial Completeness", 0.0)
    market_impact = grader_result.get("Market Impact", 0.0)
    source_quality = grader_result.get("Source Quality", 0.0)
    actionability = grader_result.get("Actionability", 0.0)
    hallucination = grader_result.get("Hallucination", 0.0)
    coherence = grader_result.get("Coherence", 0.0)
    coverage = grader_result.get("Coverage", 0.0)
    overall = grader_result.get("Overall", 0.0)

    quality = grader_result.get("Quality", "poor")
    explanation = grader_result.get("Explanation", "")

    return BloombergNewsResult(
        factual_accuracy=factual_accuracy,
        financial_completeness=financial_completeness,
        market_impact=market_impact,
        source_quality=source_quality,
        actionability=actionability,
        hallucination=hallucination,
        coherence=coherence,
        coverage=coverage,
        overall_score=overall,
        quality=NewsQuality(quality),
        explanation=explanation,
    )


async def evaluate_bloomberg_async(
    *,
    question: str,
    answer: str,
    model_config: LLMRequestConfig | None = None,
) -> BloombergNewsResult:
    """Evaluate a response as financial news content."""

    config = model_config or LLMRequestConfig()
    client_manager = AsyncClientManager.get_instance()

    user_prompt = BLOOMBERG_GRADER_PROMPT.format(
        prompt=question,
        response=answer,
    )

    try:
        completion = await run_structured_parse_call(
            openai_client=client_manager.openai_client,
            default_model=client_manager.configs.default_evaluator_model,
            model_config=config,
            system_prompt="",
            user_prompt=user_prompt,
            response_format=BloombergGraderResponse,
        )

        parsed = completion.choices[0].message.parsed

        if parsed is None:
            raise ValueError("Null grader response")

        return _parse_bloomberg_result(parsed.evaluation)

    except Exception as e:
        logger.warning(f"Bloomberg evaluation failed: {e}")
        return BloombergNewsResult(
            accuracy=0.0,
            relevance=0.0,
            insight=0.0,
            clarity=0.0,
            overall_score=0.0,
            quality=NewsQuality.POOR,
            explanation=f"Grader error: {e}",
        )


# ---------------------------------------------------------------------------
# Ground-truth grader (precision / recall / F1)
# ---------------------------------------------------------------------------

class EvaluationOutcome(str, Enum):
    """Possible outcomes for ground-truth evaluation."""

    FULLY_CORRECT = "fully_correct"
    CORRECT_WITH_EXTRANEOUS = "correct_with_extraneous"
    PARTIALLY_CORRECT = "partially_correct"
    FULLY_INCORRECT = "fully_incorrect"


class BloombergGroundTruthResult(BaseModel):
    """Result from ground-truth evaluation with IR metrics."""

    precision: float = Field(default=0.0, description="Fraction of predicted items that are correct (0-1)")
    recall: float = Field(default=0.0, description="Fraction of ground truth items that were found (0-1)")
    f1_score: float = Field(default=0.0, description="Harmonic mean of precision and recall (0-1)")
    hallucination: float = Field(default=0.0, description="Does the response avoid fabricated or unverifiable claims not supported by sources? (0-1, higher is better)")
    coherence: float = Field(default=0.0, description="Is the response logically structured, internally consistent, and easy to follow? (0-1)")
    coverage: float = Field(default=0.0, description="Does the response address all aspects of the question without omitting key topics? (0-1)")
    outcome: EvaluationOutcome = Field(default=EvaluationOutcome.FULLY_INCORRECT)
    correctness_details: dict[str, bool] = Field(default_factory=dict)
    extraneous_items: list[str] = Field(default_factory=list)
    explanation: str = Field(default="")

    def to_evaluations(self) -> list[Evaluation]:
        comment_parts = [
            f"Outcome: {self.outcome.value}",
            f"Precision: {self.precision:.2f}",
            f"Recall: {self.recall:.2f}",
            f"F1: {self.f1_score:.2f}",
            f"Hallucination: {self.hallucination:.2f}",
            f"Coherence: {self.coherence:.2f}",
            f"Coverage: {self.coverage:.2f}",
        ]
        if self.explanation:
            comment_parts.append(f"\nExplanation: {self.explanation}")
        if self.correctness_details:
            found = sum(1 for v in self.correctness_details.values() if v)
            total = len(self.correctness_details)
            comment_parts.append(f"\nCorrectness: {found}/{total} items found")
        if self.extraneous_items:
            comment_parts.append(f"\nExtraneous: {len(self.extraneous_items)} items")

        comment = "\n".join(comment_parts)

        outcome_display = {
            EvaluationOutcome.FULLY_CORRECT: "Fully Correct",
            EvaluationOutcome.CORRECT_WITH_EXTRANEOUS: "Correct with Extraneous",
            EvaluationOutcome.PARTIALLY_CORRECT: "Partially Correct",
            EvaluationOutcome.FULLY_INCORRECT: "Fully Incorrect",
        }

        return [
            Evaluation(name="Outcome", value=outcome_display.get(self.outcome, self.outcome.value), comment=self.explanation),
            Evaluation(name="F1", value=self.f1_score, comment=comment),
            Evaluation(name="Precision", value=self.precision, comment=comment),
            Evaluation(name="Recall", value=self.recall, comment=comment),
            Evaluation(name="Hallucination", value=self.hallucination, comment=comment),
            Evaluation(name="Coherence", value=self.coherence, comment=comment),
            Evaluation(name="Coverage", value=self.coverage, comment=comment),
        ]

    @staticmethod
    def error_evaluations(error_msg: str) -> list[Evaluation]:
        comment = f"Evaluation error: {error_msg}"
        return [
            Evaluation(name="Outcome", value="Fully Incorrect", comment=comment),
            Evaluation(name="F1", value=0.0, comment=comment),
            Evaluation(name="Precision", value=0.0, comment=comment),
            Evaluation(name="Recall", value=0.0, comment=comment),
            Evaluation(name="Hallucination", value=0.0, comment=comment),
            Evaluation(name="Coherence", value=0.0, comment=comment),
            Evaluation(name="Coverage", value=0.0, comment=comment),
        ]

class AnswerCorrectnessResult(BaseModel):
    explanation: str = Field(alias="Explanation", default="")
    correctness_details: dict[str, bool] = Field(alias="Correctness Details", default_factory=dict)
    excessive_answers: list[str] = Field(alias="Excessive Answers", default_factory=list)
    hallucination: float = Field(alias="Hallucination", default=0.0)
    coherence: float = Field(alias="Coherence", default=0.0)
    coverage: float = Field(alias="Coverage", default=0.0)

    model_config = {"populate_by_name": True}

class BloombergGroundTruthGraderResponse(BaseModel):
    """Structured response from the ground-truth grader."""

    answer_correctness: AnswerCorrectnessResult = Field(alias="Answer Correctness")
    #    ,description="Contains Explanation, Correctness Details, and Excessive Answers",)
    
    response_quality: dict[str, Any] = Field(
        alias="Response Quality",
        description="Contains Hallucination, Coherence, and Coverage scores",
        default_factory=dict,
    )


BLOOMBERG_GROUNDTRUTH_GRADER_PROMPT = """\
You are a senior financial analyst evaluating whether an AI assistant correctly
answered a question about Canadian bank earnings and financial news.

**Answer Correctness Task**
* **Purpose:** Assess whether the AI response provides the correct answer(s) based on
the provided "Correct Answer" and "Prompt Type".

**Financial Equivalence Rules — IMPORTANT:**
When comparing financial figures, treat these as equivalent:
- Different scales: "C$2.3 billion" = "C$2,300 million" = "$2,304 million" = "C$2.3B"
- Currency symbols: "C$" = "CAD" = "Canadian dollars"
- Percentage formats: "17 percent" = "17%" = "17 per cent"
- Minor rounding: "$2.19 billion" matches "$2.2 billion" (within 1% is acceptable)
- Implicit currency: If the question is about a Canadian bank, "$2.3 billion" without
  a currency prefix should be treated as C$ unless explicitly stated as USD.
The core financial fact matters, not the exact formatting.

**Process:**
  * Identify the "Prompt Type": "{prompt_type}".
  * Refer to the "Correct Answer": "{answer}".
  * Based on the "Prompt Type", determine if the "AI Response" contains the expected
answer(s).
    * **'Single Answer'**: Treat the correct answer as ONE answer with potentially
multiple supporting details. The key fact is the FIRST/PRIMARY claim. If the
response gets the primary fact right, mark the entire answer as correct even if
minor supporting details differ. Create only ONE key in Correctness Details
representing the core answer.
    * **'Set Answer'**: Check if the response includes *each* item from the provided
ground truth answers. The order does not matter. The response might include more
answers than the list. Determine the correctness *only* based on the list first
and then check if the response includes answers not in the list.

**Handling "Not Available" / Unanswerable Questions:**
If the correct answer states that information is "not available" or "not covered":
- The response is CORRECT if it acknowledges the information cannot be found or is
  unavailable, even if worded differently.
- The response is INCORRECT if it fabricates an answer or presents unverified claims
  as facts.

* **Explanation:** Provide a brief explanation justifying your assessment, referencing
specific parts of the AI response and the correct answer.
* **Correctness Details:** Provide a dictionary, one key for each expected answer
part, and value is a boolean indicating whether each expected answer part was found.
  * For 'Single Answer': use ONE key that captures the core expected answer.
  * For 'Set Answer': use one key per expected item.
* **Excessive Answers:** Provide a list of strings indicating any extra answer parts
in the response that are **not** in the "Correct Answer". For financial responses,
do NOT count additional context, analysis, or source citations as excessive — only
count factually distinct claims that contradict or go beyond the expected answer.
Return an empty list when there are no excessive answers.

**Response Quality Task**
Also score the AI response independently on these three dimensions (0.0 to 1.0):

* **Hallucination**: Does the response avoid fabricated or unverifiable claims not
  supported by cited sources? Score 0.0 if significant hallucinations are present,
  1.0 if every claim is grounded and verifiable.

* **Coherence**: Is the response logically structured, internally consistent, and
  easy to follow without contradictions? Score 0.0 if disjointed or contradictory,
  1.0 if clear and well-organised.

* **Coverage**: Does the response address all aspects of the question without
  omitting key topics? Score 0.0 if major aspects are ignored, 1.0 if the full
  scope of the question is addressed.

**Output Format:**
Return a valid JSON dictionary with the single top-level key "Answer Correctness".
Include the three quality scores as additional fields inside that same object.

```json
{{
  "Answer Correctness": {{
    "Explanation": "...",
    "Correctness Details": {{
      "expected_item_1": true,
      "expected_item_2": false
    }},
    "Excessive Answers": ["extra_item"],
    "Hallucination": 0.0,
    "Coherence": 0.0,
    "Coverage": 0.0
  }}
}}
```

**Now evaluate:**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>
--------------------
Rating:
"""


def _calculate_groundtruth_metrics(grader_result: AnswerCorrectnessResult) -> BloombergGroundTruthResult:
    """Calculate precision, recall, F1 from grader output."""
    #correctness_details = grader_result.get("Correctness Details", {})
    correctness_details = grader_result.correctness_details
    #extraneous_items = grader_result.get("Excessive Answers", [])
    extraneous_items = grader_result.excessive_answers
    #explanation = grader_result.get("Explanation", "")
    explanation = grader_result.explanation
    #hallucination = grader_result.get("Hallucination", 0.0)
    hallucination = grader_result.hallucination
    #coherence = grader_result.get("Coherence", 0.0)
    coherence = grader_result.coherence
    #coverage = grader_result.get("Coverage", 0.0)
    coverage = grader_result.coverage

    num_ground_truth = len(correctness_details)
    num_matched = sum(1 for v in correctness_details.values() if v)
    num_extraneous = len(extraneous_items)
    num_predicted = num_matched + num_extraneous

    precision = num_matched / num_predicted if num_predicted > 0 else 0.0
    recall = num_matched / num_ground_truth if num_ground_truth > 0 else 1.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    if num_matched == num_ground_truth and num_extraneous == 0:
        outcome = EvaluationOutcome.FULLY_CORRECT
    elif num_matched == num_ground_truth and num_extraneous > 0:
        outcome = EvaluationOutcome.CORRECT_WITH_EXTRANEOUS
    elif num_matched > 0:
        outcome = EvaluationOutcome.PARTIALLY_CORRECT
    else:
        outcome = EvaluationOutcome.FULLY_INCORRECT

    return BloombergGroundTruthResult(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        hallucination=hallucination,
        coherence=coherence,
        coverage=coverage,
        outcome=outcome,
        correctness_details=correctness_details,
        extraneous_items=extraneous_items,
        explanation=explanation,
    )


async def evaluate_bloomberg_groundtruth_async(
    *,
    question: str,
    answer: str,
    ground_truth: str,
    answer_type: str = "Single Answer",
    model_config: LLMRequestConfig | None = None,
) -> BloombergGroundTruthResult:
    """Evaluate an answer against ground truth using precision, recall, F1, hallucination, coherence, and coverage.

    Parameters
    ----------
    question : str
        The original question.
    answer : str
        The agent's answer.
    ground_truth : str
        The expected correct answer.
    answer_type : str
        "Single Answer" or "Set Answer".
    model_config : LLMRequestConfig | None
        Optional model configuration.

    Returns
    -------
    BloombergGroundTruthResult
        Evaluation result with precision, recall, F1, outcome, hallucination, coherence, and coverage.
    """
    config = model_config or LLMRequestConfig()
    client_manager = AsyncClientManager.get_instance()

    user_prompt = BLOOMBERG_GROUNDTRUTH_GRADER_PROMPT.format(
        prompt=question,
        response=answer,
        answer=ground_truth,
        prompt_type=answer_type,
    )

    try:
        completion = await run_structured_parse_call(
            openai_client=client_manager.openai_client,
            default_model=client_manager.configs.default_evaluator_model,
            model_config=config,
            system_prompt="",
            user_prompt=user_prompt,
            response_format=BloombergGroundTruthGraderResponse,
        )

        parsed = completion.choices[0].message.parsed
        logger.warning(f"RAW PARSED RESULT: {parsed.answer_correctness}")
        if parsed is None:
            raise ValueError("Null grader response")

        return _calculate_groundtruth_metrics(parsed.answer_correctness)

    except Exception as e:
        logger.warning(f"Bloomberg ground-truth evaluation failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return BloombergGroundTruthResult(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            outcome=EvaluationOutcome.FULLY_INCORRECT,
            correctness_details={},
            extraneous_items=[],
            explanation=f"Grader error: {e}",
        )
