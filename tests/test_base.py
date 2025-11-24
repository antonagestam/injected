import asyncio
import enum
from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
from operator import eq
from operator import ne
from typing import Any

import pytest
from immutables import Map

from injected import depends
from injected import resolver
from injected import seed_context
from injected._base import Marker
from injected._base import Request


class TestMarker:
    @pytest.mark.parametrize("operator", (eq, ne))
    def test_raises_not_implemented_error_when_compared(
        self,
        operator: Callable[..., Any],
    ):
        request = Request(provider=lambda: None, args=(), kwargs=Map())
        marker = Marker(request=request)
        with pytest.raises(NotImplementedError):
            operator(marker, marker)
        with pytest.raises(NotImplementedError):
            operator(marker, "other")
        with pytest.raises(NotImplementedError):
            operator("other", marker)

    def test_raises_not_implemented_error_when_cast_to_str(self):
        request = Request(provider=lambda: None, args=(), kwargs=Map())
        marker = Marker(request=request)
        with pytest.raises(NotImplementedError):
            str(marker)

    def test_raises_not_implemented_error_when_cast_to_bool(self):
        request = Request(provider=lambda: None, args=(), kwargs=Map())
        marker = Marker(request=request)
        with pytest.raises(NotImplementedError):
            bool(marker)

    def test_raises_not_implemented_error_when_used_in_logical_or(self):
        request = Request(provider=lambda: None, args=(), kwargs=Map())
        marker = Marker(request=request)
        with pytest.raises(NotImplementedError):
            marker or True  # type: ignore[unreachable]  # noqa: B018

    def test_raises_not_implemented_error_when_used_in_logical_and(self):
        request = Request(provider=lambda: None, args=(), kwargs=Map())
        marker = Marker(request=request)
        with pytest.raises(NotImplementedError):
            marker and True  # type: ignore[unreachable]  # noqa: B018


class ContextEvent(enum.Enum):
    setup = enum.auto()
    teardown = enum.auto()
    dependency = enum.auto()
    usage = enum.auto()


class TestResolver:
    def test_can_resolve_simple_dependency(self):
        value = 123

        def provider() -> int:
            return value

        @resolver
        def dependent(arg: int = depends(provider)) -> int:
            return arg

        assert dependent() == value

    def test_can_resolve_simple_async_dependency(self):
        value = 123

        async def provider() -> int:
            return value

        @resolver
        def dependent(arg: int = depends(provider)) -> int:
            return arg

        assert dependent() == value

    def test_can_resolve_concurrent_async_dependencies(self):
        async def provider(arg: int) -> int:
            return arg

        @resolver
        def dependent(
            a: int = depends(provider, arg=3),
            b: int = depends(provider, arg=5),
            c: int = depends(provider, arg=7),
        ) -> int:
            return a * b * c

        assert dependent() == 3 * 5 * 7

    def test_can_resolve_nested_dependency(self):
        value = 123

        def a() -> int:
            return value

        def b(provided: int = depends(a)) -> int:
            return provided + 2

        def c(provided: int = depends(a)) -> int:
            return provided + 1

        @resolver
        def d(
            b_value: int = depends(b),
            c_value: int = depends(c),
        ) -> int:
            return b_value + c_value

        assert d() == value + 1 + value + 2

    def test_can_inject_seed_context(self):
        @dataclass
        class LogicRequest:
            payload: dict[str, str]

        def get_request() -> LogicRequest:  # type: ignore[empty-body]
            ...

        @resolver
        def view(request: LogicRequest = depends(get_request)) -> str:
            return f"Hello {request.payload['username']}"

        context = {get_request: LogicRequest(payload={"username": "Squanchy"})}
        seeded = seed_context(view, context)

        assert seeded() == "Hello Squanchy"

    def test_can_resolve_async_dependency_with_nested_sync_dependency(self):
        value = 123

        def a() -> int:
            return value

        async def b(provided: int = depends(a)) -> int:
            return provided

        @resolver
        def c(provided: int = depends(b)) -> int:
            return provided

        assert value == c()

    def test_passes_provider_arguments(self):
        def a(value: int) -> int:
            return value

        def b(provided: int = depends(a, value=123)) -> int:
            return provided

        @resolver
        def c(
            nested: int = depends(b),
            plain: int = depends(a, 321),
        ) -> tuple[int, int]:
            return nested, plain

        assert c() == (123, 321)

    def test_reuses_resolved_value(self):
        count = 0

        def counter() -> int:
            nonlocal count
            count += 1
            return count

        def intermediate(value: int = depends(counter)) -> int:
            return value

        @resolver
        def dependent(
            plain: int = depends(counter),
            nested: int = depends(intermediate),
        ) -> tuple[int, int]:
            return plain, nested

        assert (1, 1) == dependent()
        assert 1 == count

    def test_reevaluates_resolved_value_with_differing_args(self):
        count = 0

        def counter(arg: object) -> int:
            nonlocal count
            count += 1
            return count

        def intermediate(value: int = depends(counter, arg=1)) -> int:
            return value

        @resolver
        def dependent(
            plain: int = depends(counter, arg=2),
            nested: int = depends(intermediate),
        ) -> set[int]:
            return {plain, nested}

        assert {1, 2} == dependent()
        assert 2 == count

    def test_skips_evaluation_when_dependency_passed_as_argument(self):
        count = 0

        def counter() -> int:
            nonlocal count
            count += 1
            return count

        def intermediate(value: int = depends(counter)) -> int:
            return value

        @resolver
        def dependent(value: int = depends(intermediate)) -> int:
            return value

        assert dependent(-17) == -17
        assert count == 0

    def test_can_depend_on_context_manager(self):
        # Test that:
        # - It's possible to depend on context managers.
        # - We only acquire the resource once, and then share it across the dependency
        #   graph.
        # - Context managers are properly torn down.
        events = []

        def top_level() -> int:
            events.append(ContextEvent.dependency)
            return 7

        @contextmanager
        def resource(tl: int = depends(top_level)) -> Iterator[int]:
            events.append(ContextEvent.setup)
            yield tl * 3
            events.append(ContextEvent.teardown)

        def intermediate(value: int = depends(resource)) -> int:
            events.append(ContextEvent.usage)
            return value * 11

        @resolver
        def dependent(
            a: int = depends(resource),
            b: int = depends(intermediate),
        ) -> int:
            events.append(ContextEvent.usage)
            return a * b * 5

        assert dependent() == 3 * 3 * 5 * 7 * 7 * 11
        assert events == [
            ContextEvent.dependency,
            ContextEvent.setup,
            ContextEvent.usage,
            ContextEvent.usage,
            ContextEvent.teardown,
        ]

    def test_can_depend_on_async_context_manager(self):
        # Test that:
        # - It's possible to depend on async context managers.
        # - We only acquire the resource once, and then share it across the dependency
        #   graph.
        # - Context managers are properly torn down.
        events = []

        async def top_level() -> int:
            events.append(ContextEvent.dependency)
            return 7

        @asynccontextmanager
        async def resource(tl: int = depends(top_level)) -> AsyncIterator[int]:
            events.append(ContextEvent.setup)
            yield tl * 3
            events.append(ContextEvent.teardown)

        def intermediate(value: int = depends(resource)) -> int:
            events.append(ContextEvent.usage)
            return value * 11

        @resolver
        def dependent(
            a: int = depends(resource),
            b: int = depends(intermediate),
        ) -> int:
            events.append(ContextEvent.usage)
            return a * b * 5

        assert dependent() == 3 * 3 * 5 * 7 * 7 * 11
        assert events == [
            ContextEvent.dependency,
            ContextEvent.setup,
            ContextEvent.usage,
            ContextEvent.usage,
            ContextEvent.teardown,
        ]


class TestAsyncResolver:
    async def test_can_resolve_simple_dependency(self):
        value = 123

        async def provider() -> int:
            return value

        @resolver
        async def dependent(arg: int = depends(provider)) -> int:
            return arg

        assert value == await dependent()

    async def test_can_resolve_simple_sync_dependency(self):
        value = 123

        def provider() -> int:
            return value

        @resolver
        async def dependent(arg: int = depends(provider)) -> int:
            return arg

        assert value == await dependent()

    async def test_can_resolve_nested_dependency(self):
        value = 123

        async def a() -> int:
            return value

        async def b(provided: int = depends(a)) -> int:
            return provided

        @resolver
        async def c(provided: int = depends(b)) -> int:
            return provided

        assert value == await c()

    async def test_can_resolve_nested_sync_dependency(self):
        value = 123

        def a() -> int:
            return value

        async def b(provided: int = depends(a)) -> int:
            return provided

        @resolver
        async def c(provided: int = depends(b)) -> int:
            return provided

        assert value == await c()

    async def test_can_inject_seed_context(self):
        @dataclass
        class Request:
            payload: dict[str, str]

        def get_request() -> Request:  # type: ignore[empty-body]
            ...

        async def view(request: Request = depends(get_request)) -> str:
            return f"Hello {request.payload['username']}"

        context = {get_request: Request(payload={"username": "Squanchy"})}
        seeded = seed_context(resolver(view), context)

        assert "Hello Squanchy" == await seeded()

    async def test_can_resolve_sync_dependency_with_nested_async_dependency(self):
        value = 123

        async def a() -> int:
            return value

        def b(provided: int = depends(a)) -> int:
            return provided

        @resolver
        async def c(provided: int = depends(b)) -> int:
            return provided

        assert value == await c()

    async def test_passes_provider_arguments(self):
        async def a(value: int) -> int:
            return value

        async def b(provided: int = depends(a, value=123)) -> int:
            return provided

        @resolver
        async def c(
            nested: int = depends(b),
            plain: int = depends(a, 321),
        ) -> tuple[int, int]:
            return nested, plain

        assert await c() == (123, 321)

    async def test_reuses_resolved_value(self):
        count = 0

        async def counter() -> int:
            nonlocal count
            count += 1
            return count

        async def intermediate(value: int = depends(counter)) -> int:
            return value

        @resolver
        async def dependent(
            plain: int = depends(counter),
            nested: int = depends(intermediate),
        ) -> tuple[int, int]:
            return plain, nested

        assert (1, 1) == await dependent()
        assert 1 == count

    async def test_reevaluates_resolved_value_with_differing_args(self):
        count = 0

        async def counter(arg: object) -> int:
            nonlocal count
            count += 1
            return count

        async def intermediate(value: int = depends(counter, arg=1)) -> int:
            return value

        @resolver
        async def dependent(
            plain: int = depends(counter, arg=2),
            nested: int = depends(intermediate),
        ) -> set[int]:
            return {plain, nested}

        assert {1, 2} == await dependent()
        assert 2 == count

    async def test_skips_evaluation_when_dependency_passed_as_argument(self):
        count = 0

        async def counter() -> int:
            nonlocal count
            count += 1
            return count

        async def intermediate(value: int = depends(counter)) -> int:
            return value

        @resolver
        async def dependent(value: int = depends(intermediate)) -> int:
            return value

        assert await dependent(-17) == -17
        assert count == 0

    async def test_can_resolve_concurrent_async_dependencies(self):
        async def provider(arg: int) -> int:
            return arg

        @resolver
        async def dependent(
            a: int = depends(provider, arg=3),
            b: int = depends(provider, arg=5),
            c: int = depends(provider, arg=7),
        ) -> int:
            return a * b * c

        assert await dependent() == 3 * 5 * 7

    async def test_can_resolve_slow_async_dependencies(self):
        async def provider(arg: int) -> int:
            await asyncio.sleep(arg * 0.05)
            return arg

        @resolver
        async def dependent(
            a: int = depends(provider, arg=3),
            b: int = depends(provider, arg=5),
            c: int = depends(provider, arg=7),
        ) -> int:
            return a * b * c

        assert await dependent() == 3 * 5 * 7

    async def test_can_depend_on_context_manager(self):
        # Test that:
        # - It's possible to depend on context managers.
        # - We only acquire the resource once, and then share it across the dependency
        #   graph.
        # - Context managers are properly torn down.
        events = []

        def top_level() -> int:
            events.append(ContextEvent.dependency)
            return 7

        @contextmanager
        def resource(tl: int = depends(top_level)) -> Iterator[int]:
            events.append(ContextEvent.setup)
            yield tl * 3
            events.append(ContextEvent.teardown)

        def intermediate(value: int = depends(resource)) -> int:
            events.append(ContextEvent.usage)
            return value * 11

        @resolver
        async def dependent(
            a: int = depends(resource),
            b: int = depends(intermediate),
        ) -> int:
            events.append(ContextEvent.usage)
            return a * b * 5

        assert await dependent() == 3 * 3 * 5 * 7 * 7 * 11
        assert events == [
            ContextEvent.dependency,
            ContextEvent.setup,
            ContextEvent.usage,
            ContextEvent.usage,
            ContextEvent.teardown,
        ]

    async def test_can_depend_on_async_context_manager(self):
        # Test that:
        # - It's possible to depend on async context managers.
        # - We only acquire the resource once, and then share it across the dependency
        #   graph.
        # - Context managers are properly torn down.
        events = []

        async def top_level() -> int:
            events.append(ContextEvent.dependency)
            return 7

        @asynccontextmanager
        async def resource(tl: int = depends(top_level)) -> AsyncIterator[int]:
            events.append(ContextEvent.setup)
            yield tl * 3
            events.append(ContextEvent.teardown)

        async def intermediate(value: int = depends(resource)) -> int:
            events.append(ContextEvent.usage)
            return value * 11

        @resolver
        async def dependent(
            a: int = depends(resource),
            b: int = depends(intermediate),
        ) -> int:
            events.append(ContextEvent.usage)
            return a * b * 5

        assert await dependent() == 3 * 3 * 5 * 7 * 7 * 11
        assert events == [
            ContextEvent.dependency,
            ContextEvent.setup,
            ContextEvent.usage,
            ContextEvent.usage,
            ContextEvent.teardown,
        ]
