"""ADK discovery entrypoint for the Knowledge QA agent.

Exposes a module-level ``root_agent`` so ``adk web`` can discover it.

Examples
--------
Run with ``adk web``:
    uv run adk web --port 8000 --reload --reload_agents implementations/
"""

import logging

from knowledge_qa_cibc.agent import KnowledgeGroundedAgent


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ADK discovery expects a module-level `root_agent`
root_agent = KnowledgeGroundedAgent().adk_agent
