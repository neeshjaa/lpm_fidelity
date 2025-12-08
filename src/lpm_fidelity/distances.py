import itertools
from enum import IntEnum
from functools import partial

import jax.numpy as jnp
import polars as pl
from jax import Array, jit
from jax.lax import cond, switch
from jax.scipy.special import kl_div
from jaxtyping import Integer
from scipy.spatial.distance import jensenshannon as js
from scipy.stats import entropy as scipy_entropy

from lpm_fidelity.counting import (
    OrdinalDF,
    harmonize_categorical_probabilities,
    normalize_count,
    normalize_count_bivariate_memoized,
)


class DistanceMetric(IntEnum):
    """JAX-compatible integer enum for distance metrics."""

    TVD = 0
    KL = 1
    JS = 2


def _to_distance_metric(metric) -> DistanceMetric:
    """Convert string or DistanceMetric to DistanceMetric enum."""
    if isinstance(metric, DistanceMetric):
        return metric
    if isinstance(metric, str):
        metric_map = {
            "tvd": DistanceMetric.TVD,
            "kl": DistanceMetric.KL,
            "js": DistanceMetric.JS,
        }
        return metric_map[metric.lower()]
    raise ValueError(f"Unknown distance metric: {metric}")


@jit
def fasttvd(P, Q):
    return 0.5 * jnp.sum(jnp.abs(P - Q))


@jit
def fastkl(P, Q):
    return jnp.sum(kl_div(P, Q))


@jit
def fastjs(P, Q):
    # Jensen-Shannon divergence: sqrt(0.5 * (KL(P||M) + KL(Q||M))) where M = 0.5 * (P + Q)
    M = 0.5 * (P + Q)
    return jnp.sqrt(0.5 * (jnp.sum(kl_div(P, M)) + jnp.sum(kl_div(Q, M))))


def tvd(P, Q):
    """
    Compute total variation distance between two probability vectors.

    Parameters:
    - P:  list of probabilities
    - Q:  list of probabilities

    Returns:
        Total variation distance.

    Examples:
    >>> tvd([0.5, 0.5], [0.9, 0.1])
        0.4
    """
    assert len(P) > 0
    assert len(P) == len(Q)
    return float(0.5 * sum([jnp.abs(p - q) for p, q in zip(P, Q)]))


def _distance_from_maps(ps_a, ps_b, distance_metric, overlap_required=True):
    # If we don't require overlap between columns, return 0 if one map is empty
    if not overlap_required:
        if (not ps_a) or (not ps_b):
            return None
    ps_a, ps_b = harmonize_categorical_probabilities(ps_a, ps_b)
    # The previous line ensures that the keys are the same. So the following
    # is safe to do.
    P = [ps_a[k] for k in ps_a.keys()]
    Q = [ps_b[k] for k in ps_a.keys()]
    if distance_metric == "tvd":
        return tvd(P, Q)
    elif distance_metric == "kl":
        # If qk is not null, scipy_entropy computes KL.
        return scipy_entropy(P, qk=Q)
    elif distance_metric == "js":
        return js(P, Q)
    else:
        return ValueError(f"Unknown distance metric: {distance_metric}")


def univariate_distance(column_a, column_b, distance_metric="tvd"):
    """
    Compute a set of distance metric for a pair of columns

    Parameters:
    - column_a (List or Polars Series): first column used in distance.
    - column_b (List or Polars Series): second column used in distance.
    - distance_metric (str): Choose a distance metric. One of
                              "tvd", "kl", "js".

    Returns:
        A dict with distance metric and the columns names

    Examples:
    >>> univariate_distance(
            pl.Series("foo", ["a", "b", "a", "a"]),
            pl.Series("foo", ["a", "b", "b", "b"]),
            distance_metric="tvd"
            )
        0.5
    >>> univariate_distance(
            ["a", "b", "a", "a"],
            ["a", "b", "b", "b"],
            distance_metric="tvd"
            )
        0.5
    """
    ps_a = normalize_count(column_a)
    ps_b = normalize_count(column_b)
    return _distance_from_maps(ps_a, ps_b, distance_metric)


def univariate_distances_in_data(df_a, df_b, distance_metric="tvd"):
    """
    Take two dataframes and compare a distance metric
    for all categorical_columns.

    Parameters:
    - df_a:  Polars Dataframe
    - df_b:  Polars Dataframe
    - distance_metric (str): Choose a distance metric. One of
                              "tvd", "kl", "js".

    Returns:
        A Polars Dataframe with a column "column" recording columns names
        and the distance metric used.

    Examples:
    >>> univariate_distances_in_data(df_a, df_b)
        ┌────────┬─────┐
        │ column ┆ tvd │
        │ ---    ┆ --- │
        │ str    ┆ f64 │
        ╞════════╪═════╡
        │ foo    ┆ 0.1 │
        │ bar    ┆ 0.2 │
        │ ...    ┆ ... │
        │ baz    ┆ 0.3 │
        └────────┴─────┘
       (Above is using examples values for the distance metric tvd)
    """
    assert set(df_a.columns) == set(df_b.columns)
    result = [
        {
            "column": c,
            distance_metric: univariate_distance(
                df_a[c], df_b[c], distance_metric=distance_metric
            ),
        }
        for c in df_a.columns
    ]
    return pl.DataFrame(result).sort(distance_metric, descending=False)


def _fast_distance(ps_a, ps_b, distance_metric: DistanceMetric):
    return switch(
        distance_metric,
        [
            fasttvd,
            fastkl,
            fastjs,
        ],
        ps_a,
        ps_b,
    )


@partial(jit, static_argnums=(2, 3, 4))
def bivariate_distance(
    columns_a: Integer[Array, "n 2"],
    columns_b: Integer[Array, "n 2"],
    c1_uniq_vals: int,
    c2_uniq_vals: int,
    distance_metric: DistanceMetric = DistanceMetric.TVD,
):
    """
    Compute a set of distance metric for a pair of columns

    Parameters:i
    - column_a_1 (List or Polars Series):  A column in dataframe a
    - column_a_2 (List or Polars Series):  Another column in dataframe a
    - column_b_1 (List or Polars Series):  A column in dataframe b
    - column_b_2 (List or Polars Series):  Another column in dataframe b
    - distance_metric (DistanceMetric): Choose a distance metric. One of
                              DistanceMetric.TVD, DistanceMetric.KL, DistanceMetric.JS.
    - overlap_required bool:  If  two columns don't have non-null overlap,
                              throw error

    Returns:
        A dict with a distance metric and both columns names

    Examples:
    >>> bivariate_distance(
            pl.Series("foo", ["a", "b", "a", "a"]),
            pl.Series("bar", ["x", "y", "y", "y"]),
            pl.Series("foo", ["a", "b", "a", "a"]),
            pl.Series("bar", ["x", "y", "y", "y"]),
            distance_metric="tvd"
            )
        0.0

    >>> bivariate_distance(
            pl.Series("foo", ["a", "b", "a", "a"]),
            pl.Series("bar", ["x", "y", "x", "x"]),
            pl.Series("foo", ["a", "a", "a", "b"]),
            pl.Series("bar", ["x", "x", "x", "y"]),
            distance_metric="tvd"
            )
        0.5
    """
    cs_a, a = normalize_count_bivariate_memoized(columns_a, c1_uniq_vals, c2_uniq_vals)
    cs_b, b = normalize_count_bivariate_memoized(columns_b, c1_uniq_vals, c2_uniq_vals)

    ps_a = jnp.ravel(cs_a / a)
    ps_b = jnp.ravel(cs_b / b)

    return cond(
        (a > 0) * (b > 0),
        _fast_distance,
        lambda _ps_a, _ps_b, _dm: jnp.nan,
        ps_a,
        ps_b,
        distance_metric,
    )


def bivariate_distances_in_data(
    df_a, df_b, distance_metric="tvd", overlap_required=True
):
    """
    Take two dataframes, create all pairs categorical columns.  For each pair,
    compute a probability vector of all possible events for this pair.
    Compare a distance metric for the probabilites of these events between
    the two dataframes.

    Parameters:
    - df_a:  Polars Dataframe
    - df_b:  Polars Dataframe
    - distance_metric (str): Choose a distance metric. One of
                              "tvd", "kl", "js".
    - overlap_required bool:  If  two columns don't have non-null overlap,
                              throw error

    Returns:
        A Polars Dataframe with two columns ("column-1", "column-2")
        recording columns names and the distance metric used.

    Examples:
    >>> bivariate_distances_in_data(df_a, df_b)
        ┌──────────┬──────────┬─────┐
        │ column-1 ┆ column-2 ┆ tvd │
        │ ---      ┆ ---      ┆ --- │
        │ str      ┆ str      ┆ f64 │
        ╞══════════╪══════════╪═════╡
        │ foo      ┆ bar      ┆ 1.0 │
        │ foo      ┆ baz      ┆ 2.0 │
        │ ...      ┆ ...      ┆ ... │
        │ bar      ┆ baz      ┆ 3.0 │
        └──────────┴──────────┴─────┘
       (Above is using examples values for the distance metric)
    """
    assert set(df_a.columns) == set(df_b.columns)

    # Don't drop nulls - they get encoded as -1 sentinel values
    # The bivariate counting function will filter out pairs containing -1
    odf_a, odf_b = OrdinalDF.from_dataframes([df_a, df_b])

    def _row(index_1, index_2):
        d = bivariate_distance(
            odf_a.data[:, [index_1, index_2]],
            odf_b.data[:, [index_1, index_2]],
            len(odf_a.encoders[index_1].categories_[0]),
            len(odf_a.encoders[index_2].categories_[0]),
            DistanceMetric[distance_metric.upper()],
        )

        # Convert NaN to None
        if jnp.isnan(d):
            d = None

        if d is None and overlap_required:
            raise ValueError("no overlap")

        return {
            "column-1": odf_a.columns[index_1],
            "column-2": odf_a.columns[index_2],
            distance_metric: d,
        }

    result = [
        _row(index_1, index_2)
        for index_1, index_2 in itertools.combinations(range(len(odf_a.columns)), 2)
    ]
    return pl.DataFrame(result).sort(distance_metric, descending=False)
