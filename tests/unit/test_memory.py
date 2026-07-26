"""Unit tests for the memory subsystem."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from eden.config.enums import MemoryKind, Role
from eden.config.schema import MemoryConfig, PathsConfig
from eden.core.types import Message
from eden.errors import (
    MemoryCapacityError,
    MemoryStorageError,
    MemorySubsystemError,
    ValidationError,
)
from eden.memory.conversation import ConversationMemory, ProjectMemory
from eden.memory.manager import MemoryManager, build_memory_manager
from eden.memory.repository import InMemoryRecordRepository, JsonlRecordRepository
from eden.memory.stores import LongTermMemory, ShortTermMemory
from eden.memory.types import (
    MemoryQuery,
    MemoryRecord,
    cosine_similarity,
    keyword_score,
)
from eden.memory.vector import BruteForceIndex, HashEmbedder, VectorMemory


def record(content: str, **kwargs: object) -> MemoryRecord:
    """Build a memory record for tests."""
    return MemoryRecord(content=content, **kwargs)  # type: ignore[arg-type]


class TestMemoryRecord:
    def test_identifier_is_generated(self) -> None:
        assert record("x").id.startswith("mem-")

    def test_empty_content_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            record("   ")

    def test_importance_bounds_are_enforced(self) -> None:
        with pytest.raises(ValidationError):
            record("x", importance=1.5)

    def test_empty_namespace_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            record("x", namespace=" ")

    def test_naive_timestamps_are_made_aware(self) -> None:
        stored = record("x", created_at=datetime(2026, 1, 1))
        assert stored.created_at.tzinfo is UTC

    def test_round_trips_through_its_stored_form(self) -> None:
        original = record(
            "hello",
            kind=MemoryKind.VECTOR,
            namespace="proj",
            tags=frozenset({"a", "b"}),
            metadata={"k": "v"},
        ).with_embedding([0.1, 0.2])
        restored = MemoryRecord.from_dict(original.to_dict())
        assert restored == original

    def test_malformed_stored_form_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MemoryRecord.from_dict({"id": "x"})

    def test_turn_helper_records_the_role(self) -> None:
        turn = MemoryRecord.from_turn(Role.ASSISTANT, "hi")
        assert turn.kind is MemoryKind.CONVERSATION
        assert turn.metadata["role"] == "assistant"


class TestScoringPrimitives:
    def test_cosine_of_identical_vectors_is_one(self) -> None:
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_cosine_of_orthogonal_vectors_is_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_magnitude_does_not_divide_by_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_widths_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_keyword_overlap_is_a_fraction_of_the_query(self) -> None:
        assert keyword_score(frozenset({"a", "b"}), "a c") == 0.5

    def test_empty_query_matches_everything(self) -> None:
        assert keyword_score(frozenset(), "anything") == 1.0


class TestMemoryQuery:
    def test_non_positive_limit_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MemoryQuery(limit=0)

    def test_empty_kind_set_admits_every_kind(self) -> None:
        assert MemoryQuery().matches_kind(MemoryKind.VECTOR) is True

    def test_kind_restriction_is_honoured(self) -> None:
        query = MemoryQuery(kinds=frozenset({MemoryKind.PROJECT}))
        assert query.matches_kind(MemoryKind.VECTOR) is False


class TestShortTermMemory:
    async def test_evicts_the_least_important_not_the_oldest(self) -> None:
        store = ShortTermMemory(MemoryConfig(short_term_capacity=2))
        await store.add(record("critical", importance=0.9))
        await store.add(record("trivia", importance=0.1))
        await store.add(record("newest", importance=0.5))

        contents = {hit.record.content for hit in await store.search(MemoryQuery(limit=10))}
        assert "critical" in contents
        assert "trivia" not in contents

    async def test_expired_records_are_dropped(self) -> None:
        store = ShortTermMemory(MemoryConfig(short_term_ttl_seconds=60.0))
        stale = record("old", created_at=datetime.now(tz=UTC) - timedelta(hours=2))
        await store.add(stale)
        await store.add(record("fresh"))
        assert await store.count("default") == 1

    async def test_namespaces_are_isolated(self) -> None:
        store = ShortTermMemory(MemoryConfig())
        await store.add(record("a", namespace="one"))
        await store.add(record("b", namespace="two"))
        assert await store.count("one") == 1
        assert await store.count("two") == 1

    async def test_delete_and_clear(self) -> None:
        store = ShortTermMemory(MemoryConfig())
        stored = await store.add(record("gone"))
        assert await store.delete(stored.id, namespace="default") is True
        assert await store.delete(stored.id, namespace="default") is False
        await store.add(record("x"))
        assert await store.clear("default") == 1

    async def test_search_ranks_lexical_overlap_first(self) -> None:
        store = ShortTermMemory(MemoryConfig())
        await store.add(record("the deployment pipeline failed"))
        await store.add(record("lunch was pleasant"))
        hits = await store.search(MemoryQuery(text="deployment pipeline", limit=1))
        assert hits[0].record.content.startswith("the deployment")

    async def test_tag_filter_is_a_hard_constraint(self) -> None:
        store = ShortTermMemory(MemoryConfig())
        await store.add(record("tagged", tags=frozenset({"urgent"})))
        await store.add(record("untagged"))
        hits = await store.search(MemoryQuery(tags=frozenset({"urgent"})))
        assert len(hits) == 1

    async def test_since_filter_excludes_older_records(self) -> None:
        store = ShortTermMemory(MemoryConfig())
        await store.add(record("old", created_at=datetime.now(tz=UTC) - timedelta(minutes=5)))
        await store.add(record("new"))
        cutoff = datetime.now(tz=UTC) - timedelta(minutes=1)
        assert len(await store.search(MemoryQuery(since=cutoff))) == 1

    async def test_lifecycle_is_idempotent(self) -> None:
        store = ShortTermMemory(MemoryConfig())
        await store.start()
        await store.start()
        await store.stop()
        await store.stop()


class TestRepositories:
    async def test_in_memory_round_trip(self) -> None:
        repo = InMemoryRecordRepository()
        stored = record("hello")
        await repo.append(stored)
        assert [item.id for item in await repo.read("default")] == [stored.id]
        assert await repo.remove(stored.id, "default") is True
        assert await repo.remove(stored.id, "default") is False

    async def test_jsonl_survives_a_reload(self, tmp_path: Path) -> None:
        repo = JsonlRecordRepository(tmp_path)
        await repo.append(record("persisted"))
        reopened = JsonlRecordRepository(tmp_path)
        assert [item.content for item in await reopened.read("default")] == ["persisted"]

    async def test_jsonl_skips_corrupt_lines_instead_of_failing(self, tmp_path: Path) -> None:
        repo = JsonlRecordRepository(tmp_path)
        await repo.append(record("good"))
        path = repo.path_for("default")
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{not json\n")
            handle.write('{"id": "x"}\n')
        assert [item.content for item in await repo.read("default")] == ["good"]

    async def test_jsonl_compacts_on_delete(self, tmp_path: Path) -> None:
        repo = JsonlRecordRepository(tmp_path)
        keep = await _append(repo, record("keep"))
        drop = await _append(repo, record("drop"))
        assert await repo.remove(drop.id, "default") is True
        remaining = await repo.read("default")
        assert [item.id for item in remaining] == [keep.id]

    async def test_namespace_cannot_escape_the_directory(self, tmp_path: Path) -> None:
        repo = JsonlRecordRepository(tmp_path)
        path = repo.path_for("../../etc/passwd")
        assert path.parent == tmp_path
        assert ".." not in path.name

    async def test_unreadable_directory_raises_a_storage_error(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        repo = JsonlRecordRepository(blocker)
        with pytest.raises(MemoryStorageError):
            await repo.append(record("x"))


async def _append(repo: JsonlRecordRepository, item: MemoryRecord) -> MemoryRecord:
    """Append and return the record, for readability in tests."""
    await repo.append(item)
    return item


class TestLongTermMemory:
    async def test_persists_through_the_repository(self, tmp_path: Path) -> None:
        repo = JsonlRecordRepository(tmp_path)
        store = LongTermMemory(MemoryConfig(), repo)
        await store.add(record("durable"))
        reloaded = LongTermMemory(MemoryConfig(), JsonlRecordRepository(tmp_path))
        assert await reloaded.count("default") == 1

    async def test_stamps_records_with_its_own_kind(self) -> None:
        store = LongTermMemory(MemoryConfig(), InMemoryRecordRepository())
        stored = await store.add(record("x", kind=MemoryKind.SHORT_TERM))
        assert stored.kind is MemoryKind.LONG_TERM

    async def test_ceiling_refuses_rather_than_silently_evicting(self) -> None:
        store = LongTermMemory(MemoryConfig(long_term_max_records=1), InMemoryRecordRepository())
        await store.add(record("first"))
        with pytest.raises(MemoryCapacityError):
            await store.add(record("second"))
        assert await store.count("default") == 1


class TestVectorMemory:
    def store(self) -> VectorMemory:
        return VectorMemory(MemoryConfig(), InMemoryRecordRepository(), HashEmbedder(64))

    async def test_embeddings_are_attached_on_write(self) -> None:
        stored = await self.store().add(record("embed me"))
        assert stored.embedding is not None
        assert len(stored.embedding) == 64

    async def test_semantically_closer_record_ranks_first(self) -> None:
        store = self.store()
        await store.add(record("deploy the payment service to production"))
        await store.add(record("the office cat needs feeding"))
        hits = await store.search(MemoryQuery(text="deploy payment production", limit=1))
        assert "payment" in hits[0].record.content

    async def test_batch_write_embeds_in_one_call(self) -> None:
        store = self.store()
        stored = await store.add_many([record("a b"), record("c d")])
        assert all(item.embedding is not None for item in stored)

    async def test_empty_query_browses_rather_than_returning_nothing(self) -> None:
        store = self.store()
        await store.add(record("anything"))
        assert len(await store.search(MemoryQuery(text=""))) == 1

    async def test_similarity_floor_discards_weak_hits(self) -> None:
        store = VectorMemory(
            MemoryConfig(vector_min_similarity=0.99),
            InMemoryRecordRepository(),
            HashEmbedder(64),
        )
        await store.add(record("completely unrelated subject matter"))
        assert await store.search(MemoryQuery(text="zzz", min_score=0.01)) == []

    async def test_index_tolerates_missing_and_mismatched_vectors(self) -> None:
        index = BruteForceIndex()
        scores = index.rank(
            [1.0, 0.0],
            [record("no vector"), record("wrong width").with_embedding([1.0, 2.0, 3.0])],
        )
        assert set(scores.values()) == {0.0}


class TestConversationMemory:
    async def test_history_is_chronological(self) -> None:
        store = ConversationMemory(MemoryConfig())
        await store.append_turn(Message.user("first"))
        await store.append_turn(Message.assistant("second"))
        assert [message.content for message in await store.history()] == ["first", "second"]

    async def test_roles_survive_the_round_trip(self) -> None:
        store = ConversationMemory(MemoryConfig())
        await store.append_turns([Message.system("s"), Message.user("u")])
        assert [message.role for message in await store.history()] == [
            Role.SYSTEM,
            Role.USER,
        ]

    async def test_window_keeps_system_turns_regardless_of_budget(self) -> None:
        store = ConversationMemory(MemoryConfig())
        await store.append_turn(Message.system("always keep me"))
        for index in range(20):
            await store.append_turn(Message.user(f"turn {index} " + "x" * 200))
        window = await store.window(token_budget=60)
        assert window[0].role is Role.SYSTEM

    async def test_window_prefers_recent_turns(self) -> None:
        store = ConversationMemory(MemoryConfig())
        for index in range(10):
            await store.append_turn(Message.user(f"turn{index} " + "y" * 100))
        window = await store.window(token_budget=80)
        assert "turn9" in window[-1].content

    async def test_turn_limit_trims_the_oldest(self) -> None:
        store = ConversationMemory(MemoryConfig(conversation_turn_limit=3))
        for index in range(5):
            await store.append_turn(Message.user(f"m{index}"))
        assert [m.content for m in await store.history()] == ["m2", "m3", "m4"]

    async def test_durable_history_reloads(self, tmp_path: Path) -> None:
        repo = JsonlRecordRepository(tmp_path)
        first = ConversationMemory(MemoryConfig(), repo)
        await first.append_turn(Message.user("remembered"))
        second = ConversationMemory(MemoryConfig(), JsonlRecordRepository(tmp_path))
        assert [m.content for m in await second.history()] == ["remembered"]


class TestProjectMemory:
    def memory(self) -> ProjectMemory:
        return ProjectMemory(MemoryConfig(), InMemoryRecordRepository())

    async def test_facts_are_stored_and_retrieved(self) -> None:
        project = self.memory()
        await project.remember_fact("deploy", "make ship", project="eden")
        assert await project.fact("deploy", project="eden") == "make ship"

    async def test_rewriting_a_fact_replaces_it(self) -> None:
        project = self.memory()
        await project.remember_fact("deploy", "old", project="eden")
        await project.remember_fact("deploy", "new", project="eden")
        assert await project.fact("deploy", project="eden") == "new"
        assert await project.facts(project="eden") == {"deploy": "new"}

    async def test_unknown_fact_returns_none(self) -> None:
        assert await self.memory().fact("absent", project="eden") is None

    async def test_forgetting_reports_whether_it_existed(self) -> None:
        project = self.memory()
        await project.remember_fact("k", "v", project="eden")
        assert await project.forget_fact("k", project="eden") is True
        assert await project.forget_fact("k", project="eden") is False

    async def test_projects_are_isolated(self) -> None:
        project = self.memory()
        await project.remember_fact("k", "alpha", project="a")
        await project.remember_fact("k", "beta", project="b")
        assert await project.fact("k", project="a") == "alpha"
        assert await project.fact("k", project="b") == "beta"


class TestMemoryManager:
    def manager(self, **overrides: object) -> MemoryManager:
        config = MemoryConfig(persist=False, **overrides)  # type: ignore[arg-type]
        return build_memory_manager(config, PathsConfig(), embedder=HashEmbedder(64))

    async def test_recall_merges_hits_from_every_store(self) -> None:
        manager = self.manager()
        await manager.start()
        await manager.remember("payment gateway timeout", kind=MemoryKind.LONG_TERM)
        await manager.remember("payment retry logic", kind=MemoryKind.VECTOR)
        await manager.remember("payment ticket open", kind=MemoryKind.SHORT_TERM)

        hits = await manager.recall(MemoryQuery(text="payment", limit=10))
        assert {hit.record.kind for hit in hits} >= {
            MemoryKind.LONG_TERM,
            MemoryKind.VECTOR,
            MemoryKind.SHORT_TERM,
        }
        await manager.stop()

    async def test_recall_can_be_restricted_to_one_store(self) -> None:
        manager = self.manager()
        await manager.remember("alpha", kind=MemoryKind.LONG_TERM)
        await manager.remember("alpha", kind=MemoryKind.VECTOR)
        hits = await manager.recall(MemoryQuery(text="alpha"), kinds=frozenset({MemoryKind.VECTOR}))
        assert all(hit.record.kind is MemoryKind.VECTOR for hit in hits)

    async def test_results_are_ordered_by_merged_score(self) -> None:
        manager = self.manager()
        await manager.remember("exact match phrase", kind=MemoryKind.VECTOR)
        await manager.remember("nothing alike", kind=MemoryKind.LONG_TERM)
        hits = await manager.recall(MemoryQuery(text="exact match phrase"))
        assert hits[0].record.content == "exact match phrase"
        assert hits == sorted(hits, key=lambda hit: -hit.score)

    async def test_a_failing_store_does_not_break_recall(self) -> None:
        manager = self.manager()
        await manager.remember("survivor", kind=MemoryKind.LONG_TERM)

        class Broken:
            kind = MemoryKind.SHORT_TERM
            component_name = "memory:broken"

            async def start(self) -> None:
                return

            async def stop(self) -> None:
                return

            async def search(self, query: MemoryQuery) -> list[object]:
                message = "store is down"
                raise RuntimeError(message)

        manager._stores.append(Broken())  # type: ignore[arg-type]
        hits = await manager.recall(MemoryQuery(text="survivor"))
        assert any(hit.record.content == "survivor" for hit in hits)

    async def test_forget_clears_every_store(self) -> None:
        manager = self.manager()
        await manager.remember("a", kind=MemoryKind.LONG_TERM)
        await manager.remember("b", kind=MemoryKind.VECTOR)
        assert await manager.forget("default") >= 2
        assert await manager.recall(MemoryQuery(text="a")) == []

    async def test_observe_records_a_conversation_turn(self) -> None:
        manager = self.manager()
        await manager.observe(Message.user("noted"))
        assert [m.content for m in await manager.conversation.history()] == ["noted"]

    async def test_unknown_kind_raises_with_the_available_set(self) -> None:
        manager = MemoryManager(MemoryConfig(), [])
        with pytest.raises(MemorySubsystemError) as caught:
            manager.store(MemoryKind.VECTOR)
        assert "available" in caught.value.context

    async def test_unconfigured_conversation_and_project_raise(self) -> None:
        manager = MemoryManager(MemoryConfig(), [])
        with pytest.raises(MemorySubsystemError):
            _ = manager.conversation
        with pytest.raises(MemorySubsystemError):
            _ = manager.project

    async def test_lifecycle_is_idempotent(self) -> None:
        manager = self.manager()
        await manager.start()
        await manager.start()
        await manager.stop()
        await manager.stop()

    async def test_persistent_build_writes_under_the_data_directory(self, tmp_path: Path) -> None:
        paths = PathsConfig(root=tmp_path, data_dir=tmp_path / "data")
        manager = build_memory_manager(MemoryConfig(persist=True), paths, embedder=HashEmbedder(32))
        await manager.start()
        await manager.remember("written to disk", kind=MemoryKind.LONG_TERM)
        await manager.stop()
        assert list((tmp_path / "data" / "memory" / "long_term").glob("*.jsonl"))
