"""Unit tests for durable fact extraction from conversation.

These exist to answer one concrete question: does EDEN still know a stated
name after the raw turn that stated it has fallen out of the conversation's
token window? The extractor tests check the pattern matching in isolation;
the manager tests check that observe() actually promotes a fact and that
facts_message() actually returns it; the CLI-shaped test at the end reproduces
the exact "tell it your name, then ask across a fresh session" scenario.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eden.config.enums import MemoryKind
from eden.config.schema import MemoryConfig, PathsConfig
from eden.core.types import Message
from eden.memory.conversation import ConversationMemory, ProjectMemory
from eden.memory.facts import (
    ExtractedFact,
    HeuristicFactExtractor,
    NullFactExtractor,
)
from eden.memory.manager import MemoryManager, build_memory_manager
from eden.memory.repository import InMemoryRecordRepository
from eden.memory.stores import ShortTermMemory
from eden.memory.vector import HashEmbedder


class TestHeuristicFactExtractor:
    def extractor(self) -> HeuristicFactExtractor:
        return HeuristicFactExtractor()

    @pytest.mark.parametrize(
        "text",
        [
            "my name is Sudharshan",
            "My Name Is Sudharshan.",
            "call me Sudharshan",
            "I'm called Sudharshan",
            "this is Sudharshan speaking",
        ],
    )
    def test_recognises_common_name_phrasings(self, text: str) -> None:
        facts = self.extractor().extract(text)
        assert ExtractedFact(key="name", value="Sudharshan") in facts

    def test_stops_at_a_trailing_clause(self) -> None:
        """'X and I live in Y' must not swallow the rest of the sentence."""
        facts = self.extractor().extract("my name is Sam and I live in Perth, by the way")
        by_key = {fact.key: fact.value for fact in facts}
        assert by_key["name"] == "Sam"
        assert by_key["location"] == "Perth"

    def test_recognises_location_phrasings(self) -> None:
        assert self.extractor().extract("I live in Berlin")[0].value == "Berlin"
        assert self.extractor().extract("I'm based in Tokyo")[0].value == "Tokyo"
        assert self.extractor().extract("I'm from Chennai")[0].value == "Chennai"

    def test_recognises_occupation_phrasings(self) -> None:
        facts = self.extractor().extract("I work as a mechanical engineer")
        assert facts[0].value == "mechanical engineer"

    def test_extracts_multiple_facts_from_one_message(self) -> None:
        facts = self.extractor().extract(
            "my name is Sudharshan, I live in Tiruppur, and I work as a founder"
        )
        keys = {fact.key for fact in facts}
        assert keys == {"name", "location", "occupation"}

    @pytest.mark.parametrize(
        "text",
        [
            "what is my name?",
            "the weather is nice today",
            "",
            "names are interesting",
        ],
    )
    def test_ordinary_text_yields_nothing(self, text: str) -> None:
        assert self.extractor().extract(text) == []

    def test_a_confident_extraction_is_exactly_one_point_zero(self) -> None:
        fact = self.extractor().extract("my name is Priya")[0]
        assert fact.confidence == 1.0

    def test_null_extractor_recognises_nothing(self) -> None:
        assert NullFactExtractor().extract("my name is Sam") == []


class TestManagerFactWiring:
    def manager(self) -> MemoryManager:
        conversation = ConversationMemory(MemoryConfig())
        project = ProjectMemory(MemoryConfig(), InMemoryRecordRepository())
        return MemoryManager(
            MemoryConfig(),
            [ShortTermMemory(MemoryConfig()), conversation, project.store],
            conversation=conversation,
            project=project,
            fact_extractor=HeuristicFactExtractor(),
        )

    async def test_observing_a_user_turn_promotes_a_stated_name(self) -> None:
        manager = self.manager()
        await manager.observe(Message.user("hi, my name is Sudharshan"))
        assert await manager.project.fact("name", project="default") == "Sudharshan"

    async def test_assistant_turns_are_not_scanned_for_facts(self) -> None:
        """The model itself saying a name must not be taken as the user's."""
        manager = self.manager()
        await manager.observe(Message.assistant("my name is EDEN"))
        assert await manager.project.fact("name", project="default") is None

    async def test_a_later_correction_overwrites_the_earlier_fact(self) -> None:
        manager = self.manager()
        await manager.observe(Message.user("my name is Sam"))
        await manager.observe(Message.user("actually, call me Samuel"))
        assert await manager.project.fact("name", project="default") == "Samuel"

    async def test_facts_message_summarises_what_is_known(self) -> None:
        manager = self.manager()
        await manager.observe(Message.user("my name is Priya"))
        message = await manager.facts_message("default")
        assert message is not None
        assert "name: Priya" in message.content

    async def test_no_facts_yields_no_message(self) -> None:
        manager = self.manager()
        assert await manager.facts_message("default") is None

    async def test_namespaces_keep_facts_separate(self) -> None:
        manager = self.manager()
        await manager.observe(Message.user("my name is Alice"), namespace="a")
        await manager.observe(Message.user("my name is Bob"), namespace="b")
        assert await manager.project.fact("name", project="a") == "Alice"
        assert await manager.project.fact("name", project="b") == "Bob"

    async def test_without_an_extractor_nothing_is_promoted(self) -> None:
        conversation = ConversationMemory(MemoryConfig())
        project = ProjectMemory(MemoryConfig(), InMemoryRecordRepository())
        manager = MemoryManager(
            MemoryConfig(),
            [conversation, project.store],
            conversation=conversation,
            project=project,
            fact_extractor=None,
        )
        await manager.observe(Message.user("my name is Sudharshan"))
        assert await manager.project.fact("name", project="default") is None

    async def test_without_project_memory_observe_still_succeeds(self) -> None:
        """A missing project store must not break recording the turn itself."""
        conversation = ConversationMemory(MemoryConfig())
        manager = MemoryManager(
            MemoryConfig(),
            [conversation],
            conversation=conversation,
            project=None,
            fact_extractor=HeuristicFactExtractor(),
        )
        record = await manager.observe(Message.user("my name is Sudharshan"))
        assert record.kind is MemoryKind.CONVERSATION

    async def test_a_broken_extractor_does_not_break_observe(self) -> None:
        class Explodes:
            def extract(self, text: str) -> list[ExtractedFact]:
                del text
                message = "extractor is broken"
                raise RuntimeError(message)

        conversation = ConversationMemory(MemoryConfig())
        project = ProjectMemory(MemoryConfig(), InMemoryRecordRepository())
        manager = MemoryManager(
            MemoryConfig(),
            [conversation, project.store],
            conversation=conversation,
            project=project,
            fact_extractor=Explodes(),
        )
        record = await manager.observe(Message.user("my name is Sudharshan"))
        assert record.kind is MemoryKind.CONVERSATION


class TestBuildMemoryManagerDefaultsToHeuristicExtraction:
    async def test_default_manager_extracts_a_name_automatically(self) -> None:
        manager = build_memory_manager(
            MemoryConfig(persist=False), PathsConfig(), embedder=HashEmbedder(32)
        )
        await manager.observe(Message.user("my name is Sudharshan"))
        assert await manager.project.fact("name", project="default") == "Sudharshan"


class TestSurvivesAFreshSessionLikeTheCli:
    """Reproduces exactly what was asked: tell it a name, restart the process
    equivalent, ask again, and the fact must still answer — even once the raw
    turn has fallen out of the conversation window."""

    async def test_name_outlives_a_full_manager_restart(self, tmp_path: Path) -> None:
        paths = PathsConfig(root=tmp_path, data_dir=tmp_path / "data")
        config = MemoryConfig(persist=True)

        first = build_memory_manager(config, paths, embedder=HashEmbedder(32))
        await first.start()
        await first.observe(Message.user("my name is Sudharshan"), namespace="cli")
        await first.stop()

        # A brand new manager, exactly as a brand new `eden chat` process
        # would construct — nothing carried over except what is on disk.
        second = build_memory_manager(config, paths, embedder=HashEmbedder(32))
        await second.start()
        message = await second.facts_message("cli")
        await second.stop()

        assert message is not None
        assert "Sudharshan" in message.content

    async def test_the_fact_survives_even_once_the_raw_turn_is_trimmed(self) -> None:
        """The whole point: a name must not depend on staying in the window."""
        manager = build_memory_manager(
            MemoryConfig(persist=False, conversation_turn_limit=3),
            PathsConfig(),
            embedder=HashEmbedder(32),
        )
        await manager.observe(Message.user("my name is Sudharshan"), namespace="cli")
        for index in range(10):
            await manager.observe(Message.user(f"unrelated message {index}"), namespace="cli")

        history = await manager.conversation.history("cli")
        assert not any("Sudharshan" in message.content for message in history)

        message = await manager.facts_message("cli")
        assert message is not None
        assert "Sudharshan" in message.content
