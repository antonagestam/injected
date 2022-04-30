import inspect
from dataclasses import dataclass
from functools import cache
from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterator
from typing import Mapping
from typing import NewType
from typing import NoReturn
from typing import ParamSpec
from typing import TypeVar
from typing import cast
from typing import final

from immutables import Map

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
def depends(provider: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
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


def call_inject(
    fn: Callable[P, T],
    context: Mapping[RequestKey, object],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
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
    bound = replaced.bind(*args, **kwargs)
    bound.apply_defaults()
    return fn(*bound.args, **bound.kwargs)


def inject(fn: Callable[P, T]) -> Callable[P, T]:
    dependencies = tuple(extract_dependencies(fn))

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        resolved = Map[RequestKey, object]()

        for dependency in dependencies:
            key = request_key(dependency.request)
            if key in resolved:
                continue
            result = call_inject(
                dependency.request.provider,
                resolved,
                *dependency.request.args,
                **dependency.request.kwargs,
            )
            resolved = resolved.set(key, result)
        return call_inject(fn, resolved, *args, **kwargs)

    return wrapper
