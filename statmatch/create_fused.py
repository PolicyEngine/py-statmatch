"""Implementation of create.fused for creating matched (synthetic) datasets."""

from typing import List, Optional
import pandas as pd


def create_fused(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    mtc_ids: pd.DataFrame,
    z_vars: List[str],
    dup_x: bool = False,
    match_vars: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create a fused (synthetic) dataset from matching results.

    This function creates a synthetic data frame after statistical matching
    of two data sources at the micro level. It "donates" variables from
    the donor dataset to the recipient dataset based on matching IDs.

    Parameters
    ----------
    data_rec : pd.DataFrame
        The recipient data set.
    data_don : pd.DataFrame
        The donor data set.
    mtc_ids : pd.DataFrame
        A DataFrame with two columns: 'rec.id' and 'don.id'.
        Each row contains the ID of a recipient record and the ID of
        its corresponding donor record. This is typically the output
        from nnd_hotdeck or similar matching functions.
    z_vars : List[str]
        Names of variables available only in data_don that should be
        "donated" to data_rec.
    dup_x : bool, default=False
        When True, the values of the matching variables in data_don
        are also "donated" to data_rec. The matching variables are
        renamed with a ".don" suffix to avoid confusion.
    match_vars : Optional[List[str]], default=None
        Names of the matching variables. Required when dup_x=True.

    Returns
    -------
    pd.DataFrame
        The recipient data frame with z_vars filled in from donor records.
        When dup_x=True, also includes the matching variables from donors
        with ".don" suffix.

    Raises
    ------
    ValueError
        If z_vars are not found in donor data.
        If dup_x=True but match_vars is not provided.
        If match_vars are not found in donor data.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from statmatch import nnd_hotdeck, create_fused
    >>>
    >>> # Create sample data
    >>> donor = pd.DataFrame({
    ...     'x1': [1.0, 2.0, 3.0],
    ...     'x2': [4.0, 5.0, 6.0],
    ...     'y': [10, 20, 30]
    ... })
    >>> recipient = pd.DataFrame({
    ...     'x1': [1.5, 2.5],
    ...     'x2': [4.5, 5.5]
    ... })
    >>>
    >>> # Perform matching
    >>> result = nnd_hotdeck(recipient, donor, match_vars=['x1', 'x2'])
    >>>
    >>> # Create fused dataset
    >>> fused = create_fused(recipient, donor, result['mtc.ids'], z_vars=['y'])

    Notes
    -----
    This is a Python port of R's StatMatch::create.fused function.
    See D'Orazio, M., Di Zio, M. and Scanu, M. (2006). Statistical Matching:
    Theory and Practice. Wiley, Chichester.
    """
    # Validate dup_x and match_vars
    if dup_x and match_vars is None:
        raise ValueError(
            "match_vars must be specified when dup_x=True"
        )

    # Validate z_vars exist in donor data
    missing_z = [v for v in z_vars if v not in data_don.columns]
    if missing_z:
        raise ValueError(
            f"Variables {missing_z} not found in donor data"
        )

    # Validate match_vars exist in donor data if specified
    if match_vars is not None:
        missing_m = [v for v in match_vars if v not in data_don.columns]
        if missing_m:
            raise ValueError(
                f"Match variables {missing_m} not found in donor data"
            )

    # Start with a copy of the recipient data
    result = data_rec.copy()

    # Get donor IDs from mtc_ids
    rec_ids = mtc_ids["rec.id"].values
    don_ids = mtc_ids["don.id"].values

    # Create a mapping from recipient ID to donor ID
    # Handle both integer indices and named indices
    rec_id_to_don_id = dict(zip(rec_ids, don_ids))

    # Donate z_vars from donor to recipient
    for z_var in z_vars:
        donated_values = []
        for rec_id in data_rec.index:
            don_id = rec_id_to_don_id[rec_id]
            donated_values.append(data_don.loc[don_id, z_var])
        result[z_var] = donated_values

    # If dup_x is True, also donate the matching variables with ".don" suffix
    if dup_x and match_vars is not None:
        for match_var in match_vars:
            donated_values = []
            for rec_id in data_rec.index:
                don_id = rec_id_to_don_id[rec_id]
                donated_values.append(data_don.loc[don_id, match_var])
            result[f"{match_var}.don"] = donated_values

    return result
