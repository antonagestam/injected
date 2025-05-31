<h1 align=center>injected</h1>

<p align=center>
    <a href=https://github.com/antonagestam/injected/actions?query=workflow%3ACI+branch%3Amain><img src=https://github.com/antonagestam/injected/actions/workflows/ci.yaml/badge.svg?branch=main alt="CI Build Status"></a>
    <a href=https://codecov.io/gh/antonagestam/injected><img src=https://codecov.io/gh/antonagestam/injected/branch/main/graph/badge.svg?token=GI8Z76HLYJ alt="Test coverage report"></a>
    <br>
    <a href=https://pypi.org/project/injected/><img src=https://img.shields.io/pypi/v/injected.svg?color=informational&label=PyPI alt="PyPI Package"></a>
    <a href=https://pypi.org/project/injected/><img src=https://img.shields.io/pypi/pyversions/injected.svg?color=informational&label=Python alt="Python versions"></a>
</p>

Simple, type-safe dependency injection in idiomatic Python, inspired by
[FastAPI][fastapi].

[fastapi]: https://fastapi.tiangolo.com/tutorial/dependencies/

#### Injecting dependencies

```python
from injected import depends, resolver


def get_a() -> int:
    return 13


def get_b() -> int:
    return 17


@resolver
def get_sum(
    a: int = depends(get_a),
    b: int = depends(get_b),
) -> int:
    return a + b


assert get_sum() == 30
```

#### Seeding the context of a resolver

It's sometimes useful to be able to provide an already resolved value, making it
available throughout the dependency graph. The canonical example of this is how FastAPI
makes things like requests and headers available to all dependencies.

To use this pattern, you specify a sentinel function, `get_global_value` in the example
below, and then map it to a resolved value in a context passed to `seed_context()`.

```python
from injected import depends, resolver, seed_context


def get_global_value() -> int: ...


@resolver
def calculate_value(a: int = depends(get_global_value)) -> int:
    return a + 13


seeded = seed_context(calculate_value, {get_global_value: 31})

assert seeded() == 44
```

#### Async dependencies and context managers

Dependencies can be any combination of async and non-async functions and context
managers. Async dependencies are resolved concurrently, and are scheduled at the optimal
time, as soon as their own dependencies are resolved.

Context managers are torn down upon exiting the entry-point function.

```python
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from injected import depends, resolver


async def get_a() -> int:
    return 13


def get_b() -> int:
    return 17


@asynccontextmanager
async def get_c() -> AsyncIterator[int]:
    yield 23


@resolver
async def get_sum_async(
    a: int = depends(get_a),
    b: int = depends(get_b),
    c: int = depends(get_c),
) -> int:
    return a + b + c


@resolver
def get_sum_sync(
    a: int = depends(get_a),
    b: int = depends(get_b),
    c: int = depends(get_c),
) -> int:
    return a + b + c


assert asyncio.run(get_sum_async()) == 53
assert get_sum_sync() == 53
```
