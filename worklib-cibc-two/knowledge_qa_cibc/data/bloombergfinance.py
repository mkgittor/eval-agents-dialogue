"""Bloomberg financial news dataset loader.

This module provides classes for loading and accessing Bloomberg eval data.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class GoldenEvalMetadata(BaseModel):
    """Metadata for a single golden evaluation example."""

    example_id: int = Field(description="Unique identifier for the evaluation example.")
    category: str = Field(description="Evaluation category, e.g. earnings, comparative.")
    answer_type: str = Field(description="Expected answer format type.")
    bank: str = Field(description="Target bank or group (e.g. RBC, all).")
    question_type: str = Field(description="Question type, e.g. single_article, multi_article.")
    source_date: str = Field(description="Date or date range associated with source article(s).")
    source_headline: str = Field(description="Source headline(s) for this evaluation example.")


class BloombergNewsExample(BaseModel):
    """A single Bloomberg golden evaluation question-answer pair."""

    input: str = Field(description="Evaluation input prompt/question.")
    expected_output: str = Field(description="Expected answer used for evaluation.")
    metadata: GoldenEvalMetadata = Field(description="Structured metadata for this eval entry.")


class BloombergFinancialNewsDataset:
    """Loader and manager for the Bloomberg Financial News dataset."""

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None
        self._examples: list[BloombergNewsExample] | None = None
        self._golden_eval_path = Path(__file__).with_name("golden_eval.json")

    def _load_data(self) -> None:
        """Load golden evaluation examples from local JSON file."""
        if self._examples is not None and self._df is not None:
            return

        logger.info(f"Loading golden eval data from {self._golden_eval_path}...")

        if not self._golden_eval_path.exists():
            raise FileNotFoundError(f"Golden eval file not found: {self._golden_eval_path}")

        with self._golden_eval_path.open("r", encoding="utf-8") as file:
            raw_data: Any = json.load(file)

        if not isinstance(raw_data, list):
            raise ValueError("Golden eval file must contain a JSON array of examples")

        self._examples = [BloombergNewsExample.model_validate(item) for item in raw_data]
        self._df = pd.json_normalize([example.model_dump() for example in self._examples])
        logger.info(f"Loaded {len(self._examples)} golden eval examples")

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return dataset as pandas DataFrame."""
        self._load_data()
        assert self._df is not None
        return self._df

    @property
    def examples(self) -> list[BloombergNewsExample]:
        """Return dataset as structured examples."""
        self._load_data()
        assert self._examples is not None
        return self._examples

    @property
    def golden_eval_examples(self) -> list[BloombergNewsExample]:
        """Return golden evaluation examples loaded from golden_eval.json."""
        self._load_data()
        assert self._examples is not None
        return self._examples

    @property
    def golden_eval_dataframe(self) -> pd.DataFrame:
        """Return golden evaluation examples as a pandas DataFrame."""
        return self.dataframe

    def __len__(self) -> int:
        """Return number of evaluation examples."""
        self._load_data()
        assert self._examples is not None
        return len(self._examples)

    def __getitem__(self, index: int) -> BloombergNewsExample:
        """Return one evaluation example by index."""
        self._load_data()
        assert self._examples is not None
        return self._examples[index]

    def get_by_category(self, category: str) -> list[BloombergNewsExample]:
        """Get all examples in a specific category.

        Parameters
        ----------
        category : str
            The problem category to filter by.

        Returns
        -------
        list[BloombergNewsExample]
            Examples matching the category.
        """
        return [ex for ex in self.examples if ex.category == category]

    def get_by_id(self, example_id: int) -> BloombergNewsExample | None:
        """Get a single example by its ID.

        Parameters
        ----------
        example_id : int
            The unique identifier of the example.

        Returns
        -------
        BloombergNewsExample or None
            The example with the given ID, or None if not found.
        """
        for ex in self.examples:
            if ex.example_id == example_id:
                return ex
        return None

    def get_by_ids(self, example_ids: list[int]) -> list[BloombergNewsExample]:
        """Get multiple examples by their IDs.

        Parameters
        ----------
        example_ids : list[int]
            List of example IDs to retrieve.

        Returns
        -------
        list[BloombergNewsExample]
            Examples matching the given IDs, in the order requested.
            Missing IDs are silently skipped.
        """
        id_to_example = {ex.example_id: ex for ex in self.examples}
        return [id_to_example[eid] for eid in example_ids if eid in id_to_example]

    def get_categories(self) -> list[str]:
        """Get all unique problem categories.

        Returns
        -------
        list[str]
            List of unique category names.
        """
        return list(self.dataframe["metadata.category"].unique())

    def sample(self, n: int = 10, random_state: int | None = None) -> list[BloombergNewsExample]:
        """Sample random evaluation examples."""
        self._load_data()
        assert self._examples is not None
        sample_size = min(n, len(self._examples))

        if random_state is not None:
            rng = random.Random(random_state)
            return rng.sample(self._examples, sample_size)

        return random.sample(self._examples, sample_size)
