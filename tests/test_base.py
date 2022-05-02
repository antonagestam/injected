from dataclasses import dataclass
from operator import eq
from operator import ne
from typing import Callable

import pytest
from immutables import Map

from injected import depends
from injected import resolver
from injected import seed_context
from injected._base import Request
from injected._errors import IllegalAsyncDependency


class TestRequest:
    @pytest.mark.parametrize("operator", (eq, ne))
    def test_raises_not_implemented_error_when_compared(self, operator: Callable):
        request = Request(provider=lambda: None, args=(), kwargs=Map())
        with pytest.raises(NotImplementedError):
            operator(request, request)
        with pytest.raises(NotImplementedError):
            operator(request, "other")
        with pytest.raises(NotImplementedError):
            operator("other", request)

    def test_raises_not_implemented_error_when_cast_to_str(self):
        request = Request(provider=lambda: None, args=(), kwargs=Map())
        with pytest.raises(NotImplementedError):
            str(request)


class TestInjected:
    def test_can_resolve_simple_dependency(self):
        value = 123

        def provider() -> int:
            return value

        @resolver
        def dependent(arg: int = depends(provider)) -> int:
            return arg

        assert dependent() == value

    def test_can_resolve_nested_dependency(self):
        value = 123

        def a() -> int:
            return value

        def b(provided: int = depends(a)) -> int:
            return provided

        @resolver
        def c(provided: int = depends(b)) -> int:
            return provided

        assert c() == value

    def test_can_inject_seed_context(self):
        @dataclass
        class Request:
            payload: dict[str, str]

        def get_request() -> Request:
            ...

        @resolver
        def view(request: Request = depends(get_request)) -> str:
            return f"Hello {request.payload['username']}"

        context = {get_request: Request(payload={"username": "Squanchy"})}
        seeded = seed_context(view, context)

        assert seeded() == "Hello Squanchy"

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
        ) -> tuple[int, int]:
            return plain, nested

        assert (1, 2) == dependent()

    # TODO
    @pytest.mark.xfail
    def test_skips_evaluation_when_dependency_passed_as_argument(self):
        count = 0

        def counter() -> int:
            nonlocal count
            count += 1
            return count

        @resolver
        def dependent(value: int = depends(counter)) -> int:
            return value

        assert dependent(17) == 17
        assert count == 0

    def test_raises_illegal_async_dependency_for_coroutine_dependency(self):
        async def provider() -> int:
            return 0

        with pytest.raises(IllegalAsyncDependency):

            @resolver
            def dependent(value: int = depends(provider)) -> int:
                return value


class TestAsyncInjected:
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

        def get_request() -> Request:
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
        ) -> tuple[int, int]:
            return plain, nested

        assert (1, 2) == await dependent()
