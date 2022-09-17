<h1 align=center>injected</h1>

<p align=center>
    <a href=https://github.com/antonagestam/injected/actions?query=workflow%3ACI+branch%3Amain><img src=https://github.com/antonagestam/injected/workflows/CI/badge.svg alt="CI Build Status"></a>
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


def test_resolves_dependencies():
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


def get_global_value() -> int:
    ...


@resolver
def calculate_value(a: int = depends(get_global_value)) -> int:
    return a + 13


seeded = seed_context(calculate_value, {get_global_value: 31})


def test_can_seed_resolver_context():
    assert seeded() == 44
```

#### Async dependencies

The `@resolver` decorator works with both async and non-async functions, with the
restriction that async dependencies can only be used with an async resolver. An async
resolver however, can resolve both async and vanilla dependencies.

```python
import asyncio
from injected import depends, resolver


async def get_a() -> int:
    return 13


def get_b() -> int:
    return 17


@resolver
async def get_sum(
    a: int = depends(get_a),
    b: int = depends(get_b),
) -> int:
    return a + b


def test_resolves_dependencies():
    assert asyncio.run(get_sum()) == 30
```
