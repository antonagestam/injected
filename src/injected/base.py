import inspect
from dataclasses import dataclass
from functools import cache
from typing import Callable
from typing import Iterator
from typing import Mapping
from typing import NoReturn
from typing import ParamSpec
from typing import TypeVar
from typing import cast
from typing import final

from immutables import Map


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class Request:
    provider: Callable

    def __eq__(self, other: object) -> NoReturn:
        # To avoid accidental use of sentinel values we disallow comparison.
        raise NotImplementedError("Request objects cannot be used for comparison.")

    def __str__(self) -> NoReturn:
        # To avoid accidental use of sentinel values we disallow casting to str.
        raise NotImplementedError(
            "Request objects do not have a string representations."
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Dependency:
    parameter: inspect.Parameter
    provider: Callable
    dependent: Callable


J = TypeVar("J")


def depends(provider: Callable[..., J]) -> J:
    # We intentionally lie here, for a good reason. The returned value is an instance of
    # Request, that we will use later to resolve the dependency of the parameter. If we
    # were to annotate the return type of this function accurately, as Request, it would
    # clash with the expected annotation of the resolved value. If, on the other hand,
    # we were to annotate the return type of this function as Any, users would not
    # receive type errors when a dependency provider has a return value that isn't
    # compatible with its parameter annotation.
    return cast(J, Request(provider=provider))


@cache
def extract_dependencies(dependent: Callable) -> Iterator[Dependency]:
    """
    Inspect the signature of the given function and recursively generate dependency
    representations for each dependent parameter in the function and in its
    dependency providers.
    This will yield dependencies in a resolvable order.
    """
    signature = inspect.signature(dependent)
    for parameter in signature.parameters.values():
        if not isinstance(parameter.default, Request):
            continue
        yield from extract_dependencies(parameter.default.provider)
        yield Dependency(
            parameter=parameter,
            provider=parameter.default.provider,
            dependent=dependent,
        )


P = ParamSpec("P")
K = TypeVar("K")


def call_inject(
    fn: Callable[P, K],
    context: Mapping[Callable, object],
    *args: P.args,
    **kwargs: P.kwargs,
) -> K:
    # Inspect the signature of the provider, and replace all defaults that are  Requests
    # with resolved values from `context`.
    signature = inspect.signature(fn)
    replaced = signature.replace(
        parameters=tuple(
            (
                parameter.replace(default=context[parameter.default.provider])
                if isinstance(parameter.default, Request)
                else parameter
            )
            for parameter in signature.parameters.values()
        )
    )
    # We bind the given args and kwargs to the signature, and apply our newly assigned
    # default values.
    bound = replaced.bind(*args, **kwargs)
    bound.apply_defaults()
    return fn(*bound.args, **bound.kwargs)


L = TypeVar("L")


def injected(fn: Callable[P, L]) -> Callable[P, L]:
    dependencies = tuple(extract_dependencies(fn))

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> L:
        resolved = Map[Callable, object]()
        for dependency in dependencies:
            if dependency.provider in resolved:
                continue
            result = call_inject(dependency.provider, resolved)
            resolved = resolved.set(dependency.provider, result)
        return call_inject(fn, resolved, *args, **kwargs)

    return wrapper
