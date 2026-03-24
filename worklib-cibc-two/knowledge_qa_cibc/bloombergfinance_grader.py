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
    """Evaluation result for financial news responses."""

    accuracy: float = Field(0.0, description="Factual correctness (0-1)")
    relevance: float = Field(0.0, description="Relevance to the question (0-1)")
    insight: float = Field(0.0, description="Depth of financial insight (0-1)")
    clarity: float = Field(0.0, description="Clarity and readability (0-1)")

    overall_score: float = Field(0.0, description="Average score (0-1)")
    quality: NewsQuality = Field(default=NewsQuality.POOR)

    explanation: str = Field(default="", description="Grader explanation")

    def to_evaluations(self) -> list[Evaluation]:
        comment = (
            f"Accuracy: {self.accuracy:.2f}\n"
            f"Relevance: {self.relevance:.2f}\n"
            f"Insight: {self.insight:.2f}\n"
            f"Clarity: {self.clarity:.2f}\n"
            f"Overall: {self.overall_score:.2f}\n\n"
            f"Explanation: {self.explanation}"
        )

        return [
            Evaluation(name="Quality", value=self.quality.value, comment=self.explanation),
            Evaluation(name="Overall", value=self.overall_score, comment=comment),
            Evaluation(name="Accuracy", value=self.accuracy, comment=comment),
            Evaluation(name="Insight", value=self.insight, comment=comment),
        ]

    @staticmethod
    def error_evaluations(error_msg: str) -> list[Evaluation]:
        comment = f"Evaluation error: {error_msg}"
        return [
            Evaluation(name="Quality", value="poor", comment=comment),
            Evaluation(name="Overall", value=0.0, comment=comment),
            Evaluation(name="Accuracy", value=0.0, comment=comment),
            Evaluation(name="Insight", value=0.0, comment=comment),
        ]


class BloombergGraderResponse(BaseModel):
    """Structured grader response."""

    evaluation: dict[str, Any] = Field(
        alias="Evaluation",
        description="Contains scores and explanation"
    )


BLOOMBERG_GRADER_PROMPT = """\
You are a financial news editor at Bloomberg.

Your task is to evaluate the quality of an AI-generated financial news response.

Focus on:

1. Accuracy (0-1)
- Are the financial facts correct?
- Are claims plausible and not misleading?

2. Relevance (0-1)
- Does the response directly answer the question?

3. Insight (0-1)
- Does it provide meaningful market insight or analysis?
- Does it explain implications (investors, economy, markets)?

4. Clarity (0-1)
- Is it well-written and easy to understand?

5. Overall Score (0-1)
- Average of the above

6. Quality Label
- excellent (>=0.85)
- good (>=0.7)
- fair (>=0.5)
- poor (<0.5)

Return JSON format:

{
  "Evaluation": {
    "Accuracy": float,
    "Relevance": float,
    "Insight": float,
    "Clarity": float,
    "Overall": float,
    "Quality": "excellent|good|fair|poor",
    "Explanation": "..."
  }
}

User Prompt:
{prompt}

AI Response:
{response}
"""


def _parse_bloomberg_result(grader_result: dict[str, Any]) -> BloombergNewsResult:
    accuracy = grader_result.get("Accuracy", 0.0)
    relevance = grader_result.get("Relevance", 0.0)
    insight = grader_result.get("Insight", 0.0)
    clarity = grader_result.get("Clarity", 0.0)
    overall = grader_result.get("Overall", 0.0)

    quality = grader_result.get("Quality", "poor")
    explanation = grader_result.get("Explanation", "")

    return BloombergNewsResult(
        accuracy=accuracy,
        relevance=relevance,
        insight=insight,
        clarity=clarity,
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
        ]

    @staticmethod
    def error_evaluations(error_msg: str) -> list[Evaluation]:
        comment = f"Evaluation error: {error_msg}"
        return [
            Evaluation(name="Outcome", value="Fully Incorrect", comment=comment),
            Evaluation(name="F1", value=0.0, comment=comment),
            Evaluation(name="Precision", value=0.0, comment=comment),
            Evaluation(name="Recall", value=0.0, comment=comment),
        ]


class BloombergGroundTruthGraderResponse(BaseModel):
    """Structured response from the ground-truth grader."""

    answer_correctness: dict[str, Any] = Field(
        alias="Answer Correctness",
        description="Contains Explanation, Correctness Details, and Excessive Answers",
    )


BLOOMBERG_GROUNDTRUTH_GRADER_PROMPT = """\
You are a senior financial analyst evaluating whether an AI assistant correctly
answered a question about Canadian bank earnings and financial news.

**Answer Correctness Task**
* **Purpose:** Assess whether the AI response provides the correct answer(s) based on
the provided "Correct Answer" and "Prompt Type".
* **Process:**
  * Identify the "Prompt Type": "{prompt_type}".
  * Refer to the "Correct Answer": "{answer}".
  * Based on the "Prompt Type", determine if the "AI Response" contains the expected
answer(s).
    * **'Single Answer'**: Check if the response provides the answer that addresses
the user's question. It does not have to match the exact wording of the provided
answer. For financial figures, accept equivalent representations (e.g., "C$2.3B"
matches "C$2.3 billion").
    * **'Set Answer'**: Check if the response includes *each* item from the provided
ground truth answers. The order does not matter. The response might include more
answers than the list. Determine the correctness *only* based on the list first
and then check if the response includes answers not in the list.
* **Explanation:** Provide a brief explanation justifying your assessment, referencing
specific parts of the AI response and the correct answer.
* **Correctness Details:** Provide a dictionary, one key for each expected answer
part, and value is a boolean indicating whether each expected answer part was found.
* **Excessive Answers:** Provide a list of strings indicating any extra answer parts
in the response that are **not** in the "Correct Answer". Return an empty list when
there are no excessive answers.

**Output Format:**
Return a valid JSON dictionary with the top-level key "Answer Correctness".

```json
{{
  "Answer Correctness": {{
    "Explanation": "...",
    "Correctness Details": {{
      "expected_item_1": true,
      "expected_item_2": false
    }},
    "Excessive Answers": ["extra_item"]
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


def _calculate_groundtruth_metrics(grader_result: dict[str, Any]) -> BloombergGroundTruthResult:
    """Calculate precision, recall, F1 from grader output."""
    correctness_details = grader_result.get("Correctness Details", {})
    extraneous_items = grader_result.get("Excessive Answers", [])
    explanation = grader_result.get("Explanation", "")

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
    """Evaluate an answer against ground truth using precision/recall/F1.

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
        Evaluation result with precision, recall, F1, and outcome.
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
        if parsed is None:
            raise ValueError("Null grader response")

        return _calculate_groundtruth_metrics(parsed.answer_correctness)

    except Exception as e:
        logger.warning(f"Bloomberg ground-truth evaluation failed: {e}")
        return BloombergGroundTruthResult(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            outcome=EvaluationOutcome.FULLY_INCORRECT,
            correctness_details={},
            extraneous_items=[],
            explanation=f"Grader error: {e}",
        )