from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Container
from collections.abc import Mapping
from collections.abc import Set
from contextlib import AbstractAsyncContextManager
from contextlib import AbstractContextManager
from contextlib import AsyncExitStack
from copy import replace
from dataclasses import dataclass
from functools import cache
from functools import partial
from functools import wraps
from graphlib import TopologicalSorter
from typing import Any
from typing import Final
from typing import NoReturn
from typing import cast
from typing import final
from typing import overload

from immutables import Map


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class Marker[**P, R]:
    request: Request[P, R]

    def __eq__(self, other: object) -> NoReturn:
        # To avoid accidental use of sentinel values we disallow comparison.
        raise NotImplementedError("Marker object cannot be used in comparison.")

    def __str__(self) -> NoReturn:
        # To avoid accidental use of sentinel values we disallow casting to str.
        raise NotImplementedError(
            "Marker object does not have a string representation."
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class Request[**P = Any, R = Any]:
    provider: Callable[P, R]
    args: tuple[object, ...]
    kwargs: Map[str, object]


# We intentionally "lie" in the return type here, for a good reason. The returned value
# is an instance of Request, that we will use later to resolve the dependency of the
# parameter. If we were to annotate the return type of this function accurately, as
# Request, it would clash with the expected annotation of the resolved value. If, on the
# other hand, we were to annotate the return type of this function as Any, users would
# not receive type errors when a dependency provider has a return value that isn't
# compatible with its parameter annotation.
@overload
def depends[T, **P](
    provider: Callable[P, Awaitable[T]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T: ...
@overload
def depends[T, **P](
    provider: Callable[P, AbstractContextManager[T]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T: ...
@overload
def depends[T, **P](
    provider: Callable[P, AbstractAsyncContextManager[T]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T: ...
@overload
def depends[T, **P](
    provider: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T: ...
def depends[T, **P](
    provider: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    marker = Marker(
        request=Request(
            provider=provider,
            args=tuple(args),
            kwargs=Map(kwargs),
        )
    )
    return cast(T, marker)


@cache
def get_signature(fn: Callable[..., object]) -> inspect.Signature:
    return inspect.signature(fn)


type Graph = Mapping[Request, Set[Request]]


def build_graph(
    request: Request,
    context: Container[Request],
) -> Graph:
    signature = get_signature(request.provider)
    # Bind args and kwargs so that we can omit modeling nodes that won't be needed.
    bound_arguments = signature.bind_partial(*request.args, **request.kwargs)
    bound_arguments.apply_defaults()

    graph = Map[Request, Set[Request]]().mutate()
    requests = frozenset({
        value.request
        for value in bound_arguments.arguments.values()
        if isinstance(value, Marker)
        if value.request not in context
    })
    for nested_request in requests:
        graph.update(build_graph(nested_request, context))

    graph.set(request, requests)
    return graph.finish()


sentinel: Final = object()


def execute_request[T](
    request: Request[Any, T],
    context: Map[Request, object],
) -> T:
    signature = get_signature(request.provider)
    params = []
    for parameter in signature.parameters.values():
        if (
            isinstance(parameter.default, Marker)
            and (context_value := context.get(parameter.default.request, sentinel))
            is not sentinel
        ):
            params.append(replace(parameter, default=context_value))
            continue

        params.append(parameter)

    signature = replace(signature, parameters=tuple(params))
    bound_arguments = signature.bind(*request.args, **request.kwargs)
    bound_arguments.apply_defaults()
    return request.provider(*bound_arguments.args, **bound_arguments.kwargs)


async def resolve[T](
    fn: Callable[..., T],
    seed: Context,
    args: tuple[object, ...],
    kwargs: Map[str, object],
) -> T:
    request = Request(provider=fn, args=args, kwargs=kwargs)
    context = Map({
        Request(provider=provider, args=(), kwargs=Map()): value
        for provider, value in seed.items()
    })

    # Remember: a single provider can have multiple nodes in the graph, since it shall
    # be called with different arguments as passed.
    graph = build_graph(request, context)
    topological_sorter = TopologicalSorter(graph)
    topological_sorter.prepare()

    pending_requests: dict[asyncio.Task[object], Request] = {}

    async with AsyncExitStack() as context_stack:
        while topological_sorter.is_active():
            # important: query new requests _before_ creating the context mutation.
            requests = tuple(topological_sorter.get_ready())
            context_update = context.mutate()

            for task, pending_request in tuple(pending_requests.items()):
                if not task.done():
                    continue
                context_update.set(pending_request, task.result())
                topological_sorter.done(pending_request)
                del pending_requests[task]

            for new_request in requests:
                result = execute_request(new_request, context)
                if inspect.iscoroutinefunction(new_request.provider):
                    task = asyncio.create_task(result)
                    pending_requests[task] = new_request
                elif isinstance(result, AbstractAsyncContextManager):
                    task = asyncio.create_task(
                        context_stack.enter_async_context(result)
                    )
                    pending_requests[task] = new_request
                else:
                    if isinstance(result, AbstractContextManager):
                        result = context_stack.enter_context(result)
                    context_update.set(new_request, result)
                    topological_sorter.done(new_request)

            context = context_update.finish()
            await asyncio.sleep(0)

    return context[request]  # type: ignore[return-value]


type Context = Mapping[Callable[..., Any], object]


def seed_context[C: Callable[..., Any]](
    wrapper: C,
    context: Context = Map(),
) -> C:
    return cast(C, partial(wrapper, __seed_context__=context))


def resolver[C: Callable[..., Any]](fn: C) -> C:
    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def wrapper(
            *args: object,
            __seed_context__: Context = Map(),
            **kwargs: object,
        ) -> object:
            return await resolve(fn, __seed_context__, args, Map(kwargs))

    else:

        @wraps(fn)
        def wrapper(
            *args: object,
            __seed_context__: Context = Map(),
            **kwargs: object,
        ) -> object:
            return asyncio.run(resolve(fn, __seed_context__, args, Map(kwargs)))

    return cast(C, wrapper)
