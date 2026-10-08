
import pandas as pd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from patsy import dmatrix, build_design_matrices
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

indicator_label_dict = {
    'perc_R_Rplus_Rx_rec': r'$\%R/R+/Rx$',
    'perc_R_rec': r'$\%R$',
    'perc_Rplus_rec': r'$\%R+$',
    'perc_Rx_rec': r'$\%Rx$',
    'perc_R_Rplus_Rx': r'$C(R/R+/Rx)$',
    'perc_R': r'$C(R)$',
    'perc_Rplus': r'$C(R+)$',
    'perc_Rx': r'$C(Rx)$',
    'num_items': r'$C(\mathrm{\#}items)$',
    'perc_comedy': r'$C(Comedy)$',
    'perc_action': r'$C(Action)$',
    'perc_drama': r'$C(Drama)$',
    'perc_romance': r'$C(Romance)$',
    'perc_hentai': r'$C(Hentai)$',
    'perc_horror': r'$C(Horror)$',
    'perc_fantasy': r'$C(Fantasy)$',
    'perc_kids': r'$C(Kids)$',
    'syn_sexual_rec': r'$syn-sexual$',
    'syn_violence_rec': r'$syn-violence$',
    'age' : '$Age$',
    'gender' : '$Gender$'
}

def susceptibility_estimation(
    df,
    characteristic_col,
    output_col,
    user_id_col="user_id",
    trait_type="categorical",
    continuous_model="linear",
    n_prediction_points=100,
    spline_df=4,
    random_seed=42,
):
    """
    Estimate susceptibility and 95% confidence intervals for a trait.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    characteristic_col : str
        Column containing the trait.

    output_col : str
        Column containing risk intensity in [0, 1].

    user_id_col : str, default="user_id"
        Column identifying users.

    trait_type : {"categorical", "continuous"}, default="categorical"
        Type of the trait.

    continuous_model : {"linear", "spline"}, default="linear"
        Model used for continuous traits.

    n_prediction_points : int, default=100
        Number of points at which susceptibility is predicted for a
        continuous trait.

    spline_df : int, default=4
        Degrees of freedom for the spline when continuous_model="spline".

    random_seed : int, default=42
        Reserved for reproducible extensions.

    Returns
    -------
    predictions : pd.DataFrame
        Predicted susceptibility and 95% CI.

    model_result : statsmodels result
        Fitted GLM result.
    """

    if trait_type not in {"categorical", "continuous"}:
        raise ValueError(
            "trait_type must be either 'categorical' or 'continuous'."
        )

    if continuous_model not in {"linear", "spline"}:
        raise ValueError(
            "continuous_model must be either 'linear' or 'spline'."
        )

    data = df[[characteristic_col, output_col, user_id_col]].copy()

    data = data.dropna(subset=[characteristic_col, output_col])

    if data.empty:
        raise ValueError(
            "No observations remain after removing missing values."
        )

    # Risk intensity must lie in [0, 1]
    if not data[output_col].between(0, 1).all():
        raise ValueError(
            f"'{output_col}' must contain values in [0, 1]."
        )


    if trait_type == "categorical":

        data["_trait"] = data[characteristic_col].astype("category")

        formula = f"{output_col} ~ C(_trait)"

        model = smf.glm(
            formula=formula,
            data=data,
            family=sm.families.Binomial(
                link=sm.families.links.Logit()
            ),
        )

        result = model.fit()

        categories = data["_trait"].cat.categories

        prediction_data = pd.DataFrame({
            "_trait": pd.Categorical(
                categories,
                categories=categories
            )
        })
        


        predictions = result.get_prediction(
            prediction_data
        ).summary_frame()

        counts = (
            data.groupby(
                "_trait",
                observed=True
            )
            .size()
            .rename("n")
            .reset_index()
        )

        output = pd.DataFrame({
            "trait_value": categories.astype(object),
            "n": counts["n"].to_numpy(),
            "susceptibility": predictions["mean"].to_numpy(),
            "ci_lower": predictions["mean_ci_lower"].to_numpy(),
            "ci_upper": predictions["mean_ci_upper"].to_numpy()
        })

        return output, result

    # ------------------------------------------------------------------
    # CONTINUOUS TRAIT
    # ------------------------------------------------------------------

    data["_trait"] = pd.to_numeric(
        data[characteristic_col],
        errors="coerce"
    )

    data = data.dropna(subset=["_trait"])

    if data.empty:
        raise ValueError(
            "No valid numeric observations for continuous trait."
        )

    if continuous_model == "linear":

        formula = f"{output_col} ~ _trait"

        model = smf.glm(
            formula=formula,
            data=data,
            family=sm.families.Binomial(
                link=sm.families.links.Logit()
            ),
        )

        result = model.fit()

        prediction_values = np.linspace(
            data["_trait"].min(),
            data["_trait"].max(),
            n_prediction_points
        )

        prediction_data = pd.DataFrame({
            "_trait": prediction_values
        })

        predictions = result.get_prediction(
            prediction_data
        ).summary_frame()


    else:

        spline = dmatrix(
            f"cr(_trait, df={spline_df})",
            data,
            return_type="dataframe",
        )

        spline_columns = [
            f"spline_{i}"
            for i in range(spline.shape[1])
        ]

        spline.columns = spline_columns

        data_model = pd.concat(
            [
                data.reset_index(drop=True),
                spline.reset_index(drop=True),
            ],
            axis=1,
        )

        formula = (
            f"{output_col} ~ "
            + " + ".join(spline_columns)
        )

        model = smf.glm(
            formula=formula,
            data=data_model,
            family=sm.families.Binomial(
                link=sm.families.links.Logit()
            ),
        )

        result = model.fit()

        prediction_values = np.linspace(
            data["_trait"].min(),
            data["_trait"].max(),
            n_prediction_points
        )

        prediction_base = pd.DataFrame({
            "_trait": prediction_values
        })

        prediction_spline = build_design_matrices(
            [spline.design_info],
            prediction_base,
        )[0]

        prediction_spline = pd.DataFrame(
            prediction_spline,
            columns=spline_columns,
        )


        prediction_data = prediction_spline

        predictions = result.get_prediction(
            prediction_data
        ).summary_frame()
        


    output = pd.DataFrame({
        "trait_value": prediction_values,
        "n": len(data),
        "susceptibility": predictions["mean"].to_numpy(),
        "ci_lower": predictions["mean_ci_lower"].to_numpy(),
        "ci_upper": predictions["mean_ci_upper"].to_numpy(),
    })

    return output, result

def intersectional_susceptibility_estimation(
    df,
    categorical_cols,
    continuous_cols,
    output_col,
    user_id_col="user_id",
    n_prediction_points=50,
    spline_df=4,
):
    """
    Estimate intersectional susceptibility for exactly two traits.

    The two input traits can be:
        - categorical + categorical
        - categorical + continuous
        - continuous + continuous

    Categorical traits are modeled using categorical indicators.
    Continuous traits are modeled using natural cubic splines.

    An interaction between the two traits is included so that the
    estimated susceptibility represents their joint effect.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    categorical_cols : list[str]
        Categorical input variables. Must contain 0, 1, or 2 variables.

    continuous_cols : list[str]
        Continuous input variables. Must contain 0, 1, or 2 variables.

    output_col : str
        Column containing risk intensity in [0, 1].

    user_id_col : str, default="user_id"
        Column identifying users.

    n_prediction_points : int, default=50
        Number of prediction points per continuous variable.

    spline_df : int, default=4
        Degrees of freedom for natural cubic splines.

    Returns
    -------
    predictions : pd.DataFrame
        Predicted susceptibility and 95% confidence intervals.

    result : statsmodels result
        Fitted GLM result.
    """

    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------

    if not isinstance(categorical_cols, (list, tuple)):
        raise TypeError(
            "categorical_cols must be a list or tuple."
        )

    if not isinstance(continuous_cols, (list, tuple)):
        raise TypeError(
            "continuous_cols must be a list or tuple."
        )

    input_cols = list(categorical_cols) + list(continuous_cols)

    if len(input_cols) != 2:
        raise ValueError(
            "Exactly two input variables are required."
        )

    if len(set(input_cols)) != 2:
        raise ValueError(
            "The two input variables must be different."
        )

    if len(categorical_cols) + len(continuous_cols) != 2:
        raise ValueError(
            "Inputs must consist of exactly two variables."
        )

    required_cols = input_cols + [
        output_col,
        user_id_col,
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}"
        )

    # ---------------------------------------------------------------
    # DATA
    # ---------------------------------------------------------------

    data = df[required_cols].copy()

    data = data.dropna(
        subset=input_cols + [output_col]
    )

    if data.empty:
        raise ValueError(
            "No observations remain after removing missing values."
        )

    if not data[output_col].between(0, 1).all():
        raise ValueError(
            f"'{output_col}' must contain values in [0, 1]."
        )

    # ---------------------------------------------------------------
    # IDENTIFY VARIABLE TYPES
    # ---------------------------------------------------------------

    categorical_cols = list(categorical_cols)
    continuous_cols = list(continuous_cols)

    # Convert categorical variables
    for col in categorical_cols:
        data[col] = data[col].astype("category")

    # Convert continuous variables
    for col in continuous_cols:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(
        subset=input_cols
    )

    if data.empty:
        raise ValueError(
            "No valid observations remain after type conversion."
        )

    # ---------------------------------------------------------------
    # CREATE MODEL TERMS
    # ---------------------------------------------------------------

    spline_designs = {}
    spline_columns = {}

    for col in continuous_cols:

        spline = dmatrix(
            f"cr(_x, df={spline_df})",
            data.rename(columns={col: "_x"}),
            return_type="dataframe",
        )

        columns = [
            f"{col}_spline_{i}"
            for i in range(spline.shape[1])
        ]

        spline.columns = columns

        spline_designs[col] = spline.design_info
        spline_columns[col] = columns

        data_model = spline.copy()

        for column in columns:
            data_model[column] = spline[column].to_numpy()

        data = pd.concat(
            [
                data.reset_index(drop=True),
                data_model.reset_index(drop=True),
            ],
            axis=1,
        )

    # ---------------------------------------------------------------
    # FORMULA
    # ---------------------------------------------------------------

    terms = []

    for col in categorical_cols:
        terms.append(f"C({col})")

    for col in continuous_cols:
        terms.append(
            "(" + " + ".join(spline_columns[col]) + ")"
        )

    # We explicitly construct the interaction between the two
    # variables. This works for all three combinations.
    col1 = input_cols[0]
    col2 = input_cols[1]

    def variable_term(col):

        if col in categorical_cols:
            return f"C({col})"

        return "(" + " + ".join(
            spline_columns[col]
        ) + ")"

    term1 = variable_term(col1)
    term2 = variable_term(col2)

    formula = (
        f"{output_col} ~ "
        f"{term1} + {term2} + {term1}:{term2}"
    )

    # ---------------------------------------------------------------
    # FIT MODEL
    # ---------------------------------------------------------------

    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Binomial(
            link=sm.families.links.Logit()
        ),
    )

    result = model.fit()

    # ---------------------------------------------------------------
    # CREATE PREDICTION GRID
    # ---------------------------------------------------------------

    prediction_values = {}

    for col in categorical_cols:

        categories = data[col].cat.categories

        prediction_values[col] = categories

    for col in continuous_cols:

        prediction_values[col] = np.linspace(
            data[col].min(),
            data[col].max(),
            n_prediction_points,
        )

    # Cartesian product of all values
    prediction_grid = pd.MultiIndex.from_product(
        [
            prediction_values[col]
            for col in input_cols
        ],
        names=input_cols,
    ).to_frame(index=False)

    # Restore categorical dtypes
    for col in categorical_cols:
        prediction_grid[col] = pd.Categorical(
            prediction_grid[col],
            categories=data[col].cat.categories,
        )

    # ---------------------------------------------------------------
    # BUILD SPLINE TERMS FOR PREDICTIONS
    # ---------------------------------------------------------------

    for col in continuous_cols:

        prediction_base = prediction_grid[[col]].rename(
            columns={col: "_x"}
        )

        prediction_spline = build_design_matrices(
            [spline_designs[col]],
            prediction_base,
        )[0]

        prediction_spline = pd.DataFrame(
            prediction_spline,
            columns=spline_columns[col],
        )

        prediction_grid = pd.concat(
            [
                prediction_grid.reset_index(drop=True),
                prediction_spline.reset_index(drop=True),
            ],
            axis=1,
        )

    # ---------------------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------------------

    predictions = result.get_prediction(
        prediction_grid
    ).summary_frame()

    output = prediction_grid[input_cols].copy()

    output["susceptibility"] = (
        predictions["mean"].to_numpy()
    )

    output["ci_lower"] = (
        predictions["mean_ci_lower"].to_numpy()
    )

    output["ci_upper"] = (
        predictions["mean_ci_upper"].to_numpy()
    )

    return output, result


def get_contiguous_intervals(x, mask):
    """
    Return contiguous x-intervals for which mask is True.
    """
    x = np.asarray(x)
    mask = np.asarray(mask, dtype=bool)

    if len(x) == 0 or not mask.any():
        return []

    intervals = []

    start = None

    for i, value in enumerate(mask):

        if value and start is None:
            start = i

        elif not value and start is not None:
            intervals.append(
                (x[start], x[i - 1])
            )
            start = None

    # Handle interval extending to final prediction point
    if start is not None:
        intervals.append(
            (x[start], x[-1])
        )

    return intervals

def assess_disproportionate_susceptibility(
    df,
    characteristic_col,
    output_col,
    user_id_col="user_id",
    trait_type="categorical",
    magnitude_criterion_sd=1,
    continuous_model="spline",
    n_prediction_points=100,
    spline_df=4,
    random_seed=42,
):
    """
    Estimate susceptibility, report baseline susceptibility and SD,
    and assess disproportionate susceptibility.

    For categorical traits:
        Prints each category, its predicted susceptibility, 95% CI,
        relative susceptibility, and whether it satisfies the
        magnitude and uncertainty-aware criteria.

    For continuous traits:
        Plots predicted susceptibility and its 95% CI across the
        observed trait range. Segments satisfying the magnitude
        criterion are highlighted, and confidence intervals satisfying
        the uncertainty-aware criterion are highlighted.

    Returns
    -------
    predictions : pd.DataFrame
        Predictions and criterion indicators.
    baseline : dict
        Baseline susceptibility and standard deviation.
    """

    x_true = df[characteristic_col]
    y_true = df[output_col]

    predictions, model = susceptibility_estimation(
        df=df,
        characteristic_col=characteristic_col,
        output_col=output_col,
        user_id_col=user_id_col,
        trait_type=trait_type,
        continuous_model=continuous_model,
        n_prediction_points=n_prediction_points,
        spline_df=spline_df,
        random_seed=random_seed,
    )


    valid_output = df[output_col].dropna()

    baseline_susceptibility = valid_output.mean()
    baseline_std = valid_output.std(ddof=1)

    threshold = baseline_susceptibility + magnitude_criterion_sd * baseline_std

    print("\n" + "=" * 70)
    print("BASELINE SUSCEPTIBILITY")
    print("=" * 70)

    print(
        f"Baseline susceptibility : {baseline_susceptibility:.4f}"
    )
    print(
        f"Standard deviation      : {baseline_std:.4f}"
    )
    print(
        f"Disproportionality threshold (+{magnitude_criterion_sd} SD): {threshold:.4f}"
    )

    # ---------------------------------------------------------------
    # Calculate relative susceptibility and criteria
    # ---------------------------------------------------------------

    predictions["relative_susceptibility"] = (
        predictions["susceptibility"]
        - baseline_susceptibility
    )

    # Criterion 1:
    # Predicted susceptibility exceeds baseline by > {magnitude_criterion_sd} SD.
    predictions["magnitude"] = (
        predictions["relative_susceptibility"]
        > magnitude_criterion_sd * baseline_std
    )

    # Criterion 2:
    # Entire lower side of 95% CI lies above baseline.
    predictions["strong"] = (
        predictions["ci_lower"]
        > baseline_susceptibility
    )

    # ---------------------------------------------------------------
    # CATEGORICAL
    # ---------------------------------------------------------------

    if trait_type == "categorical":

        print("\n" + "=" * 70)
        print("TRAIT VALUE SUSCEPTIBILITY")
        print("=" * 70)

        display = predictions.copy()

        display["criterion"] = np.select(
            [
                display["strong"] & display["magnitude"],
                display["strong"],
                display["magnitude"],
            ],
            [
                "UNCERTAINTY-AWARE + MAGNITUDE",
                "UNCERTAINTY-AWARE",
                "MAGNITUDE",
            ],
            default="NONE",
        )

        display = display[
            [
                "trait_value",
                "n",
                "susceptibility",
                "ci_lower",
                "ci_upper",
                "relative_susceptibility",
                "criterion",
            ]
        ]

        print(
            display.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        return predictions, {
            "baseline_susceptibility": baseline_susceptibility,
            "standard_deviation": baseline_std,
            "threshold": threshold,
            "model": model,
        }, None

    # ---------------------------------------------------------------
    # CONTINUOUS
    # ---------------------------------------------------------------

    if trait_type == "continuous":
        x = predictions["trait_value"].to_numpy()
        y = predictions["susceptibility"].to_numpy()
        lower = predictions["ci_lower"].to_numpy()
        upper = predictions["ci_upper"].to_numpy()
        

        magnitude = predictions["magnitude"].to_numpy()
        strong = predictions["strong"].to_numpy()
        
        
        magnitude_intervals = get_contiguous_intervals(
            x,
            magnitude
        )

        strong_intervals = get_contiguous_intervals(
            x,
            strong
        )


        AIC = model.aic
    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    # -----------------------------------------------------------
    # Baseline
    # -----------------------------------------------------------

    ax.axhline(
        baseline_susceptibility,
        color="0.45",
        linestyle="--",
        linewidth=1.3,
        label="Baseline susceptibility",
    )

    # +2 SD threshold
    ax.axhline(
        threshold,
        color="0.60",
        linestyle=":",
        linewidth=1.3,
        label=f"Baseline + {magnitude_criterion_sd} SD",
    )

    # -----------------------------------------------------------
    # Confidence interval
    # -----------------------------------------------------------


    ax.fill_between(
        x,
        lower,
        upper,
        color="0.20",
        alpha=0.10,
        label="95% CI",
    )

    # -----------------------------------------------------------
    # Main prediction line
    # -----------------------------------------------------------

    ax.plot(
        x,
        y,
        color="0.15",
        linewidth=2.5,
        label="Estimated susceptibility",
    )
    
    
    # ax.scatter(
    #     x_true,
    #     y_true,
    #     marker='x',
    #     s=10,
    #     alpha=0.2,
    #     color='#808080',
    #     label='True susceptibilities'
    # )

    # -----------------------------------------------------------
    # Highlight magnitude criterion
    #
    # Only the prediction line is highlighted where:
    # susceptibility > baseline + {magnitude_criterion_sd} SD
    # -----------------------------------------------------------

    y_magnitude = np.where(magnitude, y, np.nan)

    ax.plot(
        x,
        y_magnitude,
        color="#B23A3A",
        linewidth=3.5,
        label="Magnitude criterion",
    )

    # -----------------------------------------------------------
    # Highlight uncertainty-aware criterion
    #
    # The prediction and CI are highlighted where:
    # lower CI > baseline
    # -----------------------------------------------------------

    y_strong = np.where(strong, y, np.nan)
    lower_strong = np.where(strong, lower, np.nan)
    upper_strong = np.where(strong, upper, np.nan)

    # Uncertainty-aware criterion prediction
    # ax.plot(
    #     x,
    #     y_strong,
    #     color="#B23A3A",
    #     linewidth=3.5,
    #     label="Uncertainty-aware criterion",
    # )

    # Uncertainty-aware criterion confidence region
    ax.fill_between(
        x,
        lower_strong,
        upper_strong,
        color="#B23A3A",
        alpha=0.18,
        label="Uncertainty-aware criterion",
    )

    # -----------------------------------------------------------
    # Labels and formatting
    # -----------------------------------------------------------

    ax.set_xlabel(
        indicator_label_dict.get(characteristic_col, characteristic_col),
        fontsize=16
    )

    ax.set_ylabel(
        "Susceptibility",
        fontsize=16
    )

    ax.set_title(
        f"Susceptibility for {indicator_label_dict.get(output_col, output_col)}",
        fontsize=18
    )

    ax.set_ylim(
        0,
        min(
            1.0,
            max(
                y.max() * 1.2,
                threshold * 1.05
            )
        )
    )

    ax.tick_params(
    axis="both",
    labelsize=16,
    )


    ax.grid(
        axis="y",
        alpha=0.18,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="best",
        fontsize=16,
        
    )

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # Print summary of criterion regions
    # -----------------------------------------------------------

    # print("\n" + "=" * 70)
    # print("CONTINUOUS TRAIT ASSESSMENT")
    # print("=" * 70)

    # print(
    #     f"Baseline susceptibility : {baseline_susceptibility:.4f}"
    # )
    # print(
    #     f"Standard deviation      : {baseline_std:.4f}"
    # )
    # print(
    #     f"+{magnitude_criterion_sd} SD threshold         : {threshold:.4f}"
    # )

    # print(
    #     f"\nMagnitude criterion satisfied at "
    #     f"{magnitude.sum()} of {len(magnitude)} prediction points."
    # )

    # print(
    #     f"Uncertainty-aware criterion satisfied at "
    #     f"{strong.sum()} of {len(strong)} prediction points."
    # )
    
    print(
        f"AIC: {AIC:.2f}"
    )
        
    print("\nMagnitude criterion intervals:")
    if magnitude_intervals:
        for start, end in magnitude_intervals:
            start_idx = np.argmin(np.abs(x - start))
            end_idx = np.argmin(np.abs(x - end))

            print(f"X: {x[start_idx]:.4f} – {x[end_idx]:.4f}")
            print(f"Y: {y[start_idx]:.4f} – {y[end_idx]:.4f}")
    else:
        print("  None")


    print("\nUncertainty-aware criterion intervals:")
    if strong_intervals:
        for start, end in strong_intervals:
            start_idx = np.argmin(np.abs(x - start))
            end_idx = np.argmin(np.abs(x - end))

            print(f"X: {x[start_idx]:.4f} – {x[end_idx]:.4f}")
            print(f"Y: {y[start_idx]:.4f} – {y[end_idx]:.4f}")
    else:
        print("  None")

    # -----------------------------------------------------------
    # Maximum predicted susceptibility
    # -----------------------------------------------------------

    max_idx = np.argmax(y)
    max_x = x[max_idx]
    max_y = y[max_idx]

    print("\nMaximum predicted susceptibility:")
    print(f"    X = {max_x:.4f}")
    print(f"    Predicted Y = {max_y:.4f}")
    print(f"    Increase= {(max_y - baseline_susceptibility):.4f}; percentage: {(((max_y - baseline_susceptibility) / baseline_susceptibility) * 100):.4f}%")

    return predictions, {
        "baseline_susceptibility": baseline_susceptibility,
        "standard_deviation": baseline_std,
        "threshold": threshold,
        "model": model,
    }, fig


def assess_intersectional_susceptibility(
    df,
    characteristic_cols,
    output_col,
    characteristic_types,
    user_id_col="user_id",
    continuous_model="spline",
    n_prediction_points=50,
    spline_df=4,
    magnitude_criterion_sd=1,
    random_seed=42,
    figsize=(12, 6),
    vmin=None,
    vmax=None,
):
    """
    Estimate and visualize intersectional susceptibility for two traits.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    characteristic_cols : list[str]
        Exactly two characteristics defining the intersection.

    output_col : str
        Risk intensity in [0, 1].

    characteristic_types : list[str]
        Type of each characteristic:
        ["categorical", "categorical"],
        ["categorical", "continuous"],
        ["continuous", "categorical"], or
        ["continuous", "continuous"].

    continuous_model : {"linear", "spline"}
        Model used for continuous characteristics.

    n_prediction_points : int
        Number of prediction points for each continuous characteristic.

    spline_df : int
        Degrees of freedom for continuous splines.

    magnitude_criterion_sd : float
        Number of population SDs above baseline required for
        the magnitude criterion.

    Returns
    -------
    predictions : pd.DataFrame
        Intersectional predictions and criterion indicators.

    baseline : dict
        Baseline susceptibility, SD, threshold, and model.

    fig : matplotlib.figure.Figure
        Heatmap figure.
    """

    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------

    if len(characteristic_cols) != 2:
        raise ValueError(
            "characteristic_cols must contain exactly two variables."
        )

    if len(characteristic_types) != 2:
        raise ValueError(
            "characteristic_types must contain exactly two types."
        )

    if not all(
        t in {"categorical", "continuous"}
        for t in characteristic_types
    ):
        raise ValueError(
            "characteristic_types must contain only "
            "'categorical' or 'continuous'."
        )

    if len(set(characteristic_cols)) != 2:
        raise ValueError(
            "The two characteristics must be different."
        )

    # ---------------------------------------------------------------
    # SPLIT VARIABLES BY TYPE
    # ---------------------------------------------------------------

    categorical_cols = [
        col
        for col, typ in zip(characteristic_cols, characteristic_types)
        if typ == "categorical"
    ]

    continuous_cols = [
        col
        for col, typ in zip(characteristic_cols, characteristic_types)
        if typ == "continuous"
    ]

    # ---------------------------------------------------------------
    # ESTIMATION
    # ---------------------------------------------------------------

    predictions, model = intersectional_susceptibility_estimation(
        df=df,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        output_col=output_col,
        user_id_col=user_id_col,
        n_prediction_points=n_prediction_points,
        spline_df=spline_df,
    )

    # ---------------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------------

    valid_output = df[output_col].dropna()

    baseline_susceptibility = valid_output.mean()
    baseline_std = valid_output.std(ddof=1)

    threshold = (
        baseline_susceptibility
        + magnitude_criterion_sd * baseline_std
    )

    # ---------------------------------------------------------------
    # CRITERIA
    # ---------------------------------------------------------------

    predictions["relative_susceptibility"] = (
        predictions["susceptibility"]
        - baseline_susceptibility
    )

    predictions["magnitude"] = (
        predictions["susceptibility"]
        > threshold
    )

    predictions["strong"] = (
        predictions["ci_lower"]
        > baseline_susceptibility
    )

    predictions["criterion"] = np.select(
        [
            predictions["strong"] & predictions["magnitude"],
            predictions["strong"],
            predictions["magnitude"],
        ],
        [
            "UNCERTAINTY-AWARE + MAGNITUDE",
            "UNCERTAINTY-AWARE",
            "MAGNITUDE",
        ],
        default="NONE",
    )

    # ---------------------------------------------------------------
    # PRINT SUMMARY
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTERSECTIONAL SUSCEPTIBILITY")
    print("=" * 70)

    print(
        f"Traits                  : "
        f"{characteristic_cols[0]} × {characteristic_cols[1]}"
    )

    print(
        f"Baseline susceptibility : "
        f"{baseline_susceptibility:.4f}"
    )

    print(
        f"Standard deviation      : "
        f"{baseline_std:.4f}"
    )

    print(
        f"+{magnitude_criterion_sd} SD threshold     : "
        f"{threshold:.4f}"
    )

    print(
        f"\nMagnitude criterion satisfied at "
        f"{predictions['magnitude'].sum()} of "
        f"{len(predictions)} prediction points."
    )

    print(
        f"Uncertainty-aware criterion satisfied at "
        f"{predictions['strong'].sum()} of "
        f"{len(predictions)} prediction points."
    )

    if hasattr(model, "aic"):
        print(f"AIC                     : {model.aic:.2f}")

    # ---------------------------------------------------------------
    # HEATMAP DATA
    # ---------------------------------------------------------------

    x_col = characteristic_cols[0]
    y_col = characteristic_cols[1]

    heatmap = predictions.pivot(
        index=y_col,
        columns=x_col,
        values="susceptibility",
    )

    lower_heatmap = predictions.pivot(
        index=y_col,
        columns=x_col,
        values="ci_lower",
    )

    x_values = heatmap.columns.to_numpy()
    y_values = heatmap.index.to_numpy()

    z = heatmap.to_numpy(dtype=float)
    z_lower = lower_heatmap.to_numpy(dtype=float)

    # ---------------------------------------------------------------
    # FIGURE
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    all_categorical = (
        characteristic_types[0] == "categorical"
        and characteristic_types[1] == "categorical"
    )

    if all_categorical:

        # -----------------------------------------------------------
        # DATA / ORDER
        # -----------------------------------------------------------

        x_values = list(predictions[x_col].drop_duplicates())
        y_values = list(predictions[y_col].drop_duplicates())

        heatmap = (
            predictions
            .pivot(
                index=y_col,
                columns=x_col,
                values="susceptibility",
            )
            .reindex(
                index=y_values,
                columns=x_values,
            )
        )

        criterion_map = (
            predictions
            .pivot(
                index=y_col,
                columns=x_col,
                values="criterion",
            )
            .reindex(
                index=y_values,
                columns=x_values,
            )
        )

        data = heatmap.to_numpy(dtype=float)

        # -----------------------------------------------------------
        # HEATMAP
        # -----------------------------------------------------------


        imshow_kwargs = {
            "aspect": "auto",
            "cmap": "RdYlBu_r",
            "interpolation": "nearest",
        }

        if vmin is not None:
            imshow_kwargs["vmin"] = vmin

        if vmax is not None:
            imshow_kwargs["vmax"] = vmax

        mesh = ax.imshow(data, **imshow_kwargs)

        # -----------------------------------------------------------
        # CELL HATCHES + VALUES
        # -----------------------------------------------------------

        hatch_map = {
            "NONE": None,
            "UNCERTAINTY-AWARE": "//",
            "MAGNITUDE": "..",
            "UNCERTAINTY-AWARE + MAGNITUDE": "xx",
        }

        for i in range(len(y_values)):
            for j in range(len(x_values)):

                susceptibility = data[i, j]
                criterion = criterion_map.iloc[i, j]

                if pd.isna(susceptibility):
                    continue

                # Hatch layer
                hatch = hatch_map.get(criterion)

                if hatch:
                    rect = plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor="0.20",
                        linewidth=0,
                        hatch=hatch,
                        alpha=0.75,
                    )
                    ax.add_patch(rect)

                # Value annotation
                ax.text(
                    j,
                    i,
                    f"{susceptibility:.2f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="medium",
                    color="0.12",

                )

        # -----------------------------------------------------------
        # CELL SEPARATORS
        # -----------------------------------------------------------

        ax.set_xticks(
            np.arange(-0.5, len(x_values), 1),
            minor=True,
        )

        ax.set_yticks(
            np.arange(-0.5, len(y_values), 1),
            minor=True,
        )

        ax.grid(
            which="minor",
            color="white",
            linewidth=1.5,
            alpha=0.95,
        )

        ax.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )

        # -----------------------------------------------------------
        # AXES
        # -----------------------------------------------------------

        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels(
            x_values,
            fontsize=12,
        )

        ax.set_yticks(np.arange(len(y_values)))
        ax.set_yticklabels(
            y_values,
            fontsize=12,
        )

        ax.tick_params(
            axis="both",
            which="major",
            length=0,
            pad=7,
        )

        # -----------------------------------------------------------
        # LABELS / TITLE
        # -----------------------------------------------------------

        ax.set_xlabel(
            indicator_label_dict.get(x_col, x_col),
            fontsize=14,
            labelpad=11,
        )

        ax.set_ylabel(
            indicator_label_dict.get(y_col, y_col),
            fontsize=14,
            labelpad=11,
        )

        # ax.set_title(
        #     "Intersectional susceptibility",
        #     fontsize=17,
        #     fontweight="medium",
        #     pad=18,
        # )

        # ax.text(
        #     0,
        #     1.015,
        #     (
        #         f"{indicator_label_dict.get(x_col, x_col)} × "
        #         f"{indicator_label_dict.get(y_col, y_col)}"
        #     ),
        #     transform=ax.transAxes,
        #     fontsize=10.5,
        #     color="0.40",
        #     ha="left",
        #     va="bottom",
        # )

        # -----------------------------------------------------------
        # COLORBAR
        # -----------------------------------------------------------

        extend = "neither"

        if vmax is not None and np.nanmax(data) > vmax:
            extend = "max"

        if vmin is not None and np.nanmin(data) < vmin:
            extend = "both" if extend == "max" else "min"

        cbar = fig.colorbar(
            mesh,
            ax=ax,
            pad=0.025,
            fraction=0.045,
            extend=extend,
        )

        cbar.set_label(
            "Estimated susceptibility",
            fontsize=13,
            labelpad=9,
        )

        cbar.ax.tick_params(
            labelsize=10,
            length=3,
        )

        cbar.outline.set_visible(False)

        # -----------------------------------------------------------
        # HATCH LEGEND — BELOW, ONE ROW
        # -----------------------------------------------------------

        from matplotlib.patches import Patch

        legend_handles = [
            Patch(
                facecolor="none",
                edgecolor="0.20",
                hatch="//",
                label="Uncertainty-aware",
            ),
            Patch(
                facecolor="none",
                edgecolor="0.20",
                hatch="..",
                label="Magnitude",
            ),
            Patch(
                facecolor="none",
                edgecolor="0.20",
                hatch="xx",
                label="Both criteria",
            ),
        ]

        legend = ax.legend(
            handles=legend_handles,
            title="Criterion",
            loc="upper right",
            bbox_to_anchor=(-0.1, 1),
            ncol=1,
            frameon=False,
            fontsize=12,
            title_fontsize=14,
            handlelength=2.0,
            handleheight=1.4,
            columnspacing=2.2,
            handletextpad=0.7,
        )

        # -----------------------------------------------------------
        # CLEAN SPINES
        # -----------------------------------------------------------

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_facecolor("white")

        # Leave enough room for the bottom legend
        fig.subplots_adjust(
            bottom=0.20,
            top=0.88,
            right=0.90,
        )
        
        
    else:

        fig, ax = plt.subplots(
            figsize=figsize
        )

        mesh = ax.pcolormesh(
            x_values,
            y_values,
            z,
            shading="auto",
            cmap="RdYlBu_r",
        )

        cbar = fig.colorbar(
            mesh,
            ax=ax,
            pad=0.02,
        )

        cbar.set_label(
            "Estimated susceptibility",
            fontsize=15,
        )

        # ---------------------------------------------------------------
        # BASELINE CONTOUR
        # ---------------------------------------------------------------

        if (
            np.nanmin(z)
            <= baseline_susceptibility
            <= np.nanmax(z)
        ):

            ax.contour(
                x_values,
                y_values,
                z,
                levels=[baseline_susceptibility],
                colors="black",
                linewidths=2,
                linestyles="--",
            )

        # ---------------------------------------------------------------
        # MAGNITUDE CONTOUR
        # ---------------------------------------------------------------

        if (
            np.nanmin(z)
            <= threshold
            <= np.nanmax(z)
        ):

            ax.contour(
                x_values,
                y_values,
                z,
                levels=[threshold],
                colors="#B23A3A",
                linewidths=2,
                linestyles=":",
            )

        # ---------------------------------------------------------------
        # UNCERTAINTY-AWARE REGION
        # ---------------------------------------------------------------

        strong = (
            z_lower > baseline_susceptibility
        )

        if strong.any():

            ax.contour(
                x_values,
                y_values,
                strong.astype(float),
                levels=[0.5],
                colors="#B23A3A",
                linewidths=2.5,
            )



        # ---------------------------------------------------------------
        # LEGEND
        # ---------------------------------------------------------------

        legend_elements = [
            Line2D(
                [0], [0],
                color="black",
                linestyle="--",
                linewidth=2,
                label="Baseline susceptibility",
            ),
            Line2D(
                [0], [0],
                color="#B23A3A",
                linestyle=":",
                linewidth=2,
                label=f"Magnitude criterion (+{magnitude_criterion_sd} SD)",
            ),
            Line2D(
                [0], [0],
                color="#B23A3A",
                linestyle="-",
                linewidth=2.5,
                label="Uncertainty-aware criterion",
            ),
        ]

        ax.legend(
            handles=legend_elements,
            frameon=False,
            loc="best",
            fontsize=12,
        )
        # ---------------------------------------------------------------
        # LABELS
        # ---------------------------------------------------------------

        ax.set_xlabel(
            indicator_label_dict.get(
                x_col,
                x_col
            ),
            fontsize=16,
        )

        ax.set_ylabel(
            indicator_label_dict.get(
                y_col,
                y_col
            ),
            fontsize=16,
        )

        ax.set_title(
            f"Intersectional susceptibility for "
            f"{indicator_label_dict.get(x_col, x_col)} × "
            f"{indicator_label_dict.get(y_col, y_col)}",
            fontsize=18,
        )

        ax.tick_params(
            axis="both",
            labelsize=13,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------
    # RETURN
    # ---------------------------------------------------------------

    return predictions, {
        "baseline_susceptibility": baseline_susceptibility,
        "standard_deviation": baseline_std,
        "threshold": threshold,
        "model": model,
    }, fig