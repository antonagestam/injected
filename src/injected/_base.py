import inspect
from dataclasses import dataclass
from functools import cache
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import NewType
from typing import NoReturn
from typing import ParamSpec
from typing import TypeVar
from typing import cast
from typing import final
from typing import overload

from immutables import Map

from ._errors import IllegalAsyncDependency

P = ParamSpec("P")
T = TypeVar("T")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class Request(Generic[P]):
    provider: Callable[P, Any]
    args: tuple
    kwargs: Mapping[str, Any]

    def __eq__(self, other: object) -> NoReturn:
        # To avoid accidental use of sentinel values we disallow comparison.
        raise NotImplementedError("Request objects cannot be used for comparison.")

    def __str__(self) -> NoReturn:
        # To avoid accidental use of sentinel values we disallow casting to str.
        raise NotImplementedError(
            "Request objects do not have a string representations."
        )


RequestKey = NewType("RequestKey", tuple[Callable, tuple, Mapping])


def request_key(request: Request) -> RequestKey:
    return RequestKey((request.provider, request.args, request.kwargs))


@dataclass(frozen=True, slots=True, kw_only=True)
class Dependency(Generic[P]):
    request: Request[P]
    dependent: Callable


# We intentionally lie in the return type here, for a good reason. The returned value is
# an instance of Request, that we will use later to resolve the dependency of the
# parameter. If we were to annotate the return type of this function accurately, as
# Request, it would clash with the expected annotation of the resolved value. If, on the
# other hand, we were to annotate the return type of this function as Any, users would
# not receive type errors when a dependency provider has a return value that isn't
# compatible with its parameter annotation.
@overload
def depends(
    provider: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs
) -> T:
    ...


@overload
def depends(provider: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    ...


def depends(
    provider: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    request = Request(
        provider=provider,
        args=tuple(args),  # type: ignore[arg-type]
        kwargs=Map(kwargs),  # type: ignore[arg-type]
    )
    return cast(T, request)


@cache
def extract_dependencies(dependent: Callable) -> Iterator[Dependency]:
    """
    Inspect the signature of the given function and recursively generate dependency
    representations for each dependent parameter in the function and in its dependency
    providers. Dependencies are generated in a resolvable order.
    """
    signature = inspect.signature(dependent)
    for parameter in signature.parameters.values():
        if not isinstance(parameter.default, Request):
            continue
        yield from extract_dependencies(parameter.default.provider)
        yield Dependency(request=parameter.default, dependent=dependent)


def resolve_arguments(
    fn: Callable[P, object],
    context: Mapping[RequestKey, object],
    *args: P.args,
    **kwargs: P.kwargs,
) -> inspect.BoundArguments:
    # Inspect the signature of the provider, and replace all Request defaults with
    # resolved values from `context`.
    signature = inspect.signature(fn)
    replaced = signature.replace(
        parameters=tuple(
            (
                parameter.replace(default=context[request_key(parameter.default)])
                if isinstance(parameter.default, Request)
                else parameter
            )
            for parameter in signature.parameters.values()
        )
    )
    # Bind the given args and kwargs to the signature, and apply the newly assigned
    # default values.
    bound_arguments = replaced.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return bound_arguments


def assert_no_async_dependencies(
    fn: Callable, dependencies: Iterable[Dependency]
) -> None:
    async_dependencies = {
        dependency.request.provider.__name__
        for dependency in dependencies
        if inspect.iscoroutinefunction(dependency.request.provider)
    }
    if async_dependencies:
        raise IllegalAsyncDependency(
            f"Cannot depend on coroutine dependencies (defined with async def) "
            f"when in a synchronous context. To use async dependencies, "
            f"@{inject.__name__} must be applied to an async function. "
            f"({fn.__name__} is not async, but these dependencies are: "
            f"{async_dependencies})."
        )


def create_sync_wrapper(
    fn: Callable[P, T], dependencies: Iterable[Dependency]
) -> Callable[P, T]:
    assert_no_async_dependencies(fn, dependencies)

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        resolved = Map[RequestKey, object]()

        for dependency in dependencies:
            key = request_key(dependency.request)
            if key in resolved:
                continue
            bound_arguments = resolve_arguments(
                dependency.request.provider,
                resolved,
                *dependency.request.args,
                **dependency.request.kwargs,
            )
            result = dependency.request.provider(
                *bound_arguments.args, **bound_arguments.kwargs
            )
            resolved = resolved.set(key, result)

        bound_arguments = resolve_arguments(fn, resolved, *args, **kwargs)
        return fn(*bound_arguments.args, **bound_arguments.kwargs)

    return wrapper


def create_async_wrapper(
    fn: Callable[P, Awaitable[T]], dependencies: Iterable[Dependency]
) -> Callable[P, Awaitable[T]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        resolved = Map[RequestKey, object]()

        for dependency in dependencies:
            key = request_key(dependency.request)
            if key in resolved:
                continue
            bound_arguments = resolve_arguments(
                dependency.request.provider,
                resolved,
                *dependency.request.args,
                **dependency.request.kwargs,
            )
            task = dependency.request.provider(
                *bound_arguments.args, **bound_arguments.kwargs
            )
            result = await task if inspect.iscoroutine(task) else task
            resolved = resolved.set(key, result)

        bound_arguments = resolve_arguments(fn, resolved, *args, **kwargs)
        return await fn(*bound_arguments.args, **bound_arguments.kwargs)

    return wrapper


C = TypeVar("C", bound=Callable)


def inject(fn: C) -> C:
    dependencies = tuple(extract_dependencies(fn))
    return cast(
        C,
        create_async_wrapper(fn, dependencies)
        if inspect.iscoroutinefunction(fn)
        else create_sync_wrapper(fn, dependencies),
    )
