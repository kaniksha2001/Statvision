"""Hypothesis Testing page.

Parametric, non-parametric and categorical tests. Each test renders the null
and alternative hypotheses, statistic, p-value, decision and a plain-language
interpretation through the shared :func:`test_result_card`.
"""
from __future__ import annotations

import streamlit as st

from statvision.core.utils import (categorical_columns, clean_series,
                                    numeric_columns)
from statvision.inference import hypothesis_tests as ht
from statvision.pages_impl.components import (page_header, require_data,
                                              show_table, test_result_card)


def _group_arrays(df, value_col, group_col):
    """Split a numeric column into one array per category of ``group_col``."""
    groups, labels = [], []
    for label, sub in df.groupby(group_col)[value_col]:
        arr = clean_series(sub)
        if arr.size > 0:
            groups.append(arr)
            labels.append(str(label))
    return groups, labels


def render() -> None:
    page_header(
        "Hypothesis Testing",
        "Parametric, non-parametric and categorical significance tests.",
        help_text=("Choose a test that matches your design. **Parametric** tests "
                   "assume approximately normal data; their **non-parametric** "
                   "counterparts relax that assumption by working on ranks. The "
                   "decision compares the p-value to your chosen significance "
                   "level α."),
    )
    df = require_data()
    num = numeric_columns(df)
    cat = categorical_columns(df)

    alpha = st.selectbox("Significance α", [0.10, 0.05, 0.01], index=1,
                         key="ht_alpha")

    tab_param, tab_nonparam, tab_cat = st.tabs(
        ["Parametric", "Non-parametric", "Categorical"])

    # --- Parametric --------------------------------------------------------
    with tab_param:
        test = st.selectbox(
            "Test",
            ["One-sample t-test", "Independent t-test", "Paired t-test",
             "One-way ANOVA"], key="ht_param_test")
        try:
            if test == "One-sample t-test":
                col = st.selectbox("Variable", num, key="ht_os_col")
                popmean = st.number_input("Hypothesised mean μ₀", value=0.0,
                                          key="ht_os_mu")
                if st.button("Run test", key="ht_os_run"):
                    res = ht.one_sample_t(clean_series(df[col]), popmean, alpha)
                    test_result_card(res, report_title=f"One-sample t — {col}")

            elif test == "Independent t-test":
                c1, c2 = st.columns(2)
                val = c1.selectbox("Numeric variable", num, key="ht_it_val")
                grp = c2.selectbox("Grouping variable", cat, key="ht_it_grp")
                equal = st.checkbox("Assume equal variances (Student's)",
                                    value=False, key="ht_it_eq")
                groups, labels = _group_arrays(df, val, grp)
                if len(groups) != 2:
                    st.info("Independent t-test needs exactly two groups; "
                            f"'{grp}' has {len(groups)}.")
                elif st.button("Run test", key="ht_it_run"):
                    res = ht.independent_t(groups[0], groups[1],
                                           equal_var=equal, alpha=alpha)
                    res["interpretation"] += (
                        f" Comparing {labels[0]} vs {labels[1]}.")
                    test_result_card(res, report_title=f"Independent t — {val}")

            elif test == "Paired t-test":
                c1, c2 = st.columns(2)
                a = c1.selectbox("Variable A", num, key="ht_pt_a")
                b = c2.selectbox("Variable B", num,
                                 index=min(1, len(num) - 1), key="ht_pt_b")
                if st.button("Run test", key="ht_pt_run"):
                    pair = df[[a, b]].apply(lambda s: s).dropna()
                    res = ht.paired_t(pair[a].to_numpy(), pair[b].to_numpy(),
                                      alpha)
                    test_result_card(res, report_title=f"Paired t — {a} vs {b}")

            else:  # One-way ANOVA
                c1, c2 = st.columns(2)
                val = c1.selectbox("Numeric variable", num, key="ht_an_val")
                grp = c2.selectbox("Grouping variable", cat, key="ht_an_grp")
                groups, labels = _group_arrays(df, val, grp)
                if len(groups) < 2:
                    st.info("ANOVA needs at least two groups.")
                elif st.button("Run test", key="ht_an_run"):
                    res = ht.one_way_anova(groups, alpha)
                    res["interpretation"] += f" Groups: {', '.join(labels)}."
                    test_result_card(res, report_title=f"ANOVA — {val} by {grp}")
        except Exception as exc:
            st.error(f"Test failed: {exc}")

    # --- Non-parametric ----------------------------------------------------
    with tab_nonparam:
        test = st.selectbox(
            "Test",
            ["Mann-Whitney U", "Wilcoxon signed-rank", "Kruskal-Wallis H"],
            key="ht_np_test")
        try:
            if test == "Mann-Whitney U":
                c1, c2 = st.columns(2)
                val = c1.selectbox("Numeric variable", num, key="ht_mw_val")
                grp = c2.selectbox("Grouping variable", cat, key="ht_mw_grp")
                groups, labels = _group_arrays(df, val, grp)
                if len(groups) != 2:
                    st.info("Mann-Whitney U needs exactly two groups.")
                elif st.button("Run test", key="ht_mw_run"):
                    res = ht.mann_whitney(groups[0], groups[1], alpha)
                    res["interpretation"] += f" {labels[0]} vs {labels[1]}."
                    test_result_card(res, report_title=f"Mann-Whitney — {val}")

            elif test == "Wilcoxon signed-rank":
                c1, c2 = st.columns(2)
                a = c1.selectbox("Variable A", num, key="ht_wx_a")
                b = c2.selectbox("Variable B", num,
                                 index=min(1, len(num) - 1), key="ht_wx_b")
                if st.button("Run test", key="ht_wx_run"):
                    pair = df[[a, b]].dropna()
                    res = ht.wilcoxon_signed_rank(pair[a].to_numpy(),
                                                  pair[b].to_numpy(), alpha)
                    test_result_card(res, report_title=f"Wilcoxon — {a} vs {b}")

            else:  # Kruskal-Wallis
                c1, c2 = st.columns(2)
                val = c1.selectbox("Numeric variable", num, key="ht_kw_val")
                grp = c2.selectbox("Grouping variable", cat, key="ht_kw_grp")
                groups, labels = _group_arrays(df, val, grp)
                if len(groups) < 2:
                    st.info("Kruskal-Wallis needs at least two groups.")
                elif st.button("Run test", key="ht_kw_run"):
                    res = ht.kruskal_wallis(groups, alpha)
                    res["interpretation"] += f" Groups: {', '.join(labels)}."
                    test_result_card(res, report_title=f"Kruskal-Wallis — {val}")
        except Exception as exc:
            st.error(f"Test failed: {exc}")

    # --- Categorical -------------------------------------------------------
    with tab_cat:
        if len(cat) < 2:
            st.info("Need at least two categorical columns for a chi-square "
                    "test of independence.")
        else:
            c1, c2 = st.columns(2)
            a = c1.selectbox("Variable A", cat, key="ht_chi_a")
            b = c2.selectbox("Variable B", cat,
                             index=min(1, len(cat) - 1), key="ht_chi_b")
            if a == b:
                st.warning("Choose two different variables.")
            elif st.button("Run test", key="ht_chi_run"):
                try:
                    res = ht.chi_square_independence(df, a, b, alpha)
                    test_result_card(res,
                                     report_title=f"Chi-square — {a} × {b}")
                    st.markdown("**Contingency table (observed)**")
                    show_table(res["contingency"])
                    st.markdown("**Expected counts under independence**")
                    show_table(res["expected"].round(2))
                except Exception as exc:
                    st.error(f"Test failed: {exc}")
