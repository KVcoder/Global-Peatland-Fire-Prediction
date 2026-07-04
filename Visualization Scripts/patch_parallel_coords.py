#!/usr/bin/env python3
"""
Patch script v2 - targets the ALREADY-PATCHED version of the file.

Fixes:
  - Trial number colorbar: fontsize=18 on label and ticks
  - Best trial legend box: font back down to 13, compact size
  - All axis/tick fonts stay at 18

Usage:
    python patch_parallel_coords_v2.py <input_file> <output_file>
"""

import sys
import os
import re


def apply_patches(src: str) -> str:
    patches = []
    errors = []

    # ── helper ───────────────────────────────────────────────────────────────
    def add(old, new, name):
        patches.append((old, new, name))

    # ─────────────────────────────────────────────────────────────────────────
    # Detect which version we're working with by probing key strings
    # ─────────────────────────────────────────────────────────────────────────
    already_patched_labels = '"eta" if for_parallel else "Learning rate' in src
    original_labels = '"log10(eta)" if for_parallel else "Learning rate' in src

    print(f"  [detect] already-patched labels: {already_patched_labels}")
    print(f"  [detect] original labels:         {original_labels}")

    # =========================================================================
    # GROUP A: label patches (only needed if file still has log10() labels)
    # =========================================================================
    if original_labels:
        # PATCH A1: replot _plot_vals (4-space indent, has blank lines)
        add(
            '''    def _plot_vals(series, name, for_parallel=False):
        vals = _pd.to_numeric(series, errors="coerce").astype(float).to_numpy()

        if name in {"eta", "learning_rate", "min_child_weight", "reg_lambda", "lambda"} and _np.nanmin(vals) > 0:
            if name == "eta":
                label = "log10(eta)" if for_parallel else "Learning rate η (log10 scale)"
            elif name == "min_child_weight":
                label = "log10(min_child_weight)" if for_parallel else "Min child weight (log10 scale)"
            else:
                label = f"log10({name})"

            return _np.log10(vals), label

        if name == "eta":
            return vals, ("log10(eta)" if for_parallel else "Learning rate η")

        return vals, name

    param_pairs = _parse_pairs(replot_args.hpo_visualize_params)''',
            '''    def _plot_vals(series, name, for_parallel=False):
        vals = _pd.to_numeric(series, errors="coerce").astype(float).to_numpy()

        if name in {"eta", "learning_rate", "min_child_weight", "reg_lambda", "lambda"} and _np.nanmin(vals) > 0:
            if name == "eta":
                label = "eta" if for_parallel else "Learning rate η (log10 scale)"
            elif name == "min_child_weight":
                label = "min_child_wt" if for_parallel else "Min child weight (log10 scale)"
            else:
                label = name

            return _np.log10(vals), label

        if name == "eta":
            return vals, ("eta" if for_parallel else "Learning rate η")

        if name == "min_child_weight":
            return vals, ("min_child_wt" if for_parallel else "Min child weight")

        return vals, name

    param_pairs = _parse_pairs(replot_args.hpo_visualize_params)''',
            "A1: replot _plot_vals labels",
        )

        # PATCH A2: demo _plot_vals (4-space indent, no blank lines)
        add(
            '''    def _plot_vals(series, name, for_parallel=False):
        vals = _pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
        if name in {"eta", "learning_rate", "min_child_weight", "reg_lambda", "lambda"} and _np.nanmin(vals) > 0:
            if name == "eta":
                label = "log10(eta)" if for_parallel else "Learning rate η (log10 scale)"
            elif name == "min_child_weight":
                label = "log10(min_child_weight)" if for_parallel else "Min child weight (log10 scale)"
            else:
                label = f"log10({name})"
            return _np.log10(vals), label
        if name == "eta":
            return vals, ("log10(eta)" if for_parallel else "Learning rate η")
        return vals, name''',
            '''    def _plot_vals(series, name, for_parallel=False):
        vals = _pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
        if name in {"eta", "learning_rate", "min_child_weight", "reg_lambda", "lambda"} and _np.nanmin(vals) > 0:
            if name == "eta":
                label = "eta" if for_parallel else "Learning rate η (log10 scale)"
            elif name == "min_child_weight":
                label = "min_child_wt" if for_parallel else "Min child weight (log10 scale)"
            else:
                label = name
            return _np.log10(vals), label
        if name == "eta":
            return vals, ("eta" if for_parallel else "Learning rate η")
        if name == "min_child_weight":
            return vals, ("min_child_wt" if for_parallel else "Min child weight")
        return vals, name''',
            "A2: demo _plot_vals labels",
        )

        # PATCH A3: _hpo_param_axis_label
        add(
            '''def _hpo_param_axis_label(name: str, *, for_parallel: bool = False) -> str:
    name = str(name)
    if name == "eta":
        return "log10(eta)" if for_parallel else "Learning rate η (log10 scale)"
    if name == "min_child_weight":
        return "log10(min_child_weight)" if for_parallel else "Min child weight (log10 scale)"
    if name == "max_depth":''',
            '''def _hpo_param_axis_label(name: str, *, for_parallel: bool = False) -> str:
    name = str(name)
    if name == "eta":
        return "eta" if for_parallel else "Learning rate η (log10 scale)"
    if name == "min_child_weight":
        return "min_child_wt" if for_parallel else "Min child weight (log10 scale)"
    if name == "max_depth":''',
            "A3: _hpo_param_axis_label",
        )

        # PATCH A4: _hpo_plot_transform
        add(
            '''def _hpo_plot_transform(series: pd.Series, name: str, *, for_parallel: bool = False) -> Tuple[np.ndarray, str]:
    """Transform log-scale params for visually readable axes."""
    vals = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    log_params = {
        "eta", "learning_rate", "min_child_weight", "reg_lambda", "lambda",
        "residual_reg_lambda", "platt_reg",
    }
    if name in log_params and np.nanmin(vals) > 0:
        return np.log10(vals), f"log10({name})"
    return vals, name''',
            '''def _hpo_plot_transform(series: pd.Series, name: str, *, for_parallel: bool = False) -> Tuple[np.ndarray, str]:
    """Transform log-scale params for visually readable axes."""
    vals = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    log_params = {
        "eta", "learning_rate", "min_child_weight", "reg_lambda", "lambda",
        "residual_reg_lambda", "platt_reg",
    }
    _display = {"min_child_weight": "min_child_wt"}
    display_name = _display.get(name, name)
    if name in log_params and np.nanmin(vals) > 0:
        return np.log10(vals), display_name
    return vals, display_name''',
            "A4: _hpo_plot_transform",
        )

    else:
        print("  [skip] label patches — labels already updated")

    # =========================================================================
    # GROUP B: font / colorbar / legend size patches
    # These target whatever state the rendering blocks are currently in.
    # We use a flexible regex approach so we don't have to match the exact
    # current fontsize numbers.
    # =========================================================================

    # We'll do these with direct regex replacements rather than exact-string
    # patches, so they work regardless of whether previous patches ran.

    def re_sub_once(pattern, replacement, text, name):
        new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
        if n == 0:
            errors.append(f"  MISSING (regex): {name}")
        else:
            print(f"  OK (regex, {n} replacement): {name}")
        return new_text

    # ── B1: replot colorbar + axes + legend block ────────────────────────────
    # Matches the colorbar.set_label line (with or without fontsize arg),
    # then the xticks/tick_params/ylim/ylabel/title/grid/legend block.
    src = re_sub_once(
        r'(fig\.colorbar\(sm, ax=ax, pad=0\.02\))(\.\s*set_label\("Trial number"\)|\.set_label\("Trial number"\))\s*\n'
        r'(\s*ax\.set_xticks\(xs\)\s*\n)'
        r'(\s*ax\.set_xticklabels\(labels.*?fontsize=\d+.*?\)\s*\n)'
        r'(\s*ax\.tick_params\(axis="x".*?\)\s*\n)'
        r'((?:\s*ax\.tick_params\(axis="y".*?\)\s*\n)?)'
        r'(\s*ax\.set_ylim.*?\n)'
        r'(\s*ax\.set_ylabel.*?fontsize=\d+.*?\n)'
        r'(\s*ax\.set_title.*?fontsize=\d+.*?\n)'
        r'(\s*ax\.grid.*?\n)'
        r'\s*\n'
        r'(\s*leg = ax\.legend\(\s*\n)'
        r'(\s*loc="upper right",\s*\n)'
        r'(\s*fontsize=\d+,\s*\n)',
        lambda m: (
            '        _cbar = fig.colorbar(sm, ax=ax, pad=0.02)\n'
            '        _cbar.set_label("Trial number", fontsize=18)\n'
            '        _cbar.ax.tick_params(labelsize=18)\n'
            '\n'
            '        ax.set_xticks(xs)\n'
            '        ax.set_xticklabels(labels, rotation=14, ha="right", fontsize=18)\n'
            '        ax.tick_params(axis="x", pad=12, labelsize=18)\n'
            '        ax.tick_params(axis="y", labelsize=18)\n'
            '\n'
            '        ax.set_ylim(-0.04, 1.04)\n'
            '        ax.set_ylabel("Normalized parameter value", fontsize=18)\n'
            '        ax.set_title("HPO Parallel Coordinate Plot \u2014 high-performance corridor", fontsize=18)\n'
            '        ax.grid(True, axis="y", linestyle="--", alpha=0.25)\n'
            '\n'
            '        leg = ax.legend(\n'
            '            loc="upper right",\n'
            '            fontsize=13,\n'
        ),
        src,
        "B1: replot colorbar+axes+legend-open",
    )

    # ── B2: replot legend body (handlelength / borderpad / labelspacing) ─────
    # and frame styling — we just normalise these to the "original" values
    src = re_sub_once(
        r'(        leg\.get_frame\(\)\.set_alpha\(1\.0\)\s*\n'
        r'        leg\.get_frame\(\)\.set_facecolor\("white"\)\s*\n'
        r'        leg\.get_frame\(\)\.set_edgecolor\("black"\)\s*\n'
        r'        leg\.get_frame\(\)\.set_linewidth\()(\d+(?:\.\d+)?)'
        r'(\)\s*\n'
        r'        _plt\.setp\(leg\.get_lines\(\), linewidth=)(\d+(?:\.\d+)?)'
        r'(\)\s*\n'
        r'        leg\.set_zorder\(20\))',
        r'\g<1>1.5\g<3>5.0\g<5>',
        src,
        "B2: replot legend frame sizes",
    )

    # ── B3: replot legend handlelength/borderpad/labelspacing ────────────────
    src = re_sub_once(
        r'(        leg = ax\.legend\(\s*\n'
        r'            loc="upper right",\s*\n'
        r'            fontsize=\d+,\s*\n'
        r'            frameon=True,\s*\n'
        r'            fancybox=False,\s*\n'
        r'            framealpha=1\.0,\s*\n'
        r'            facecolor="white",\s*\n'
        r'            edgecolor="black",\s*\n'
        r'            handlelength=)(\d+(?:\.\d+)?)'
        r'(,\s*\n'
        r'            borderpad=)(\d+(?:\.\d+)?)'
        r'(,\s*\n'
        r'            labelspacing=)(\d+(?:\.\d+)?)'
        r'(,\s*\n        \)\s*\n\s*\n        leg\.get_frame)',
        r'\g<1>3.4\g<3>0.9\g<5>0.8\g<7>',
        src,
        "B3: replot legend handle/border/spacing",
    )

    # ── B4: demo axes + legend ───────────────────────────────────────────────
    # The demo block uses semicolons on same lines
    src = re_sub_once(
        r'ax\.set_xticks\(xs\); ax\.set_xticklabels\(labels, rotation=14, ha="right", fontsize=\d+\)'
        r'\s*\n(\s*)ax\.tick_params\(axis="x", pad=\d+(?:, labelsize=\d+)?\)'
        r'(?:\s*\n\s*ax\.tick_params\(axis="y".*?\))?'
        r'\s*\n(\s*)ax\.set_ylim\(-0\.04, 1\.04\); ax\.set_ylabel\("Normalized parameter value"(?:, fontsize=\d+)?\)'
        r'\s*\n(\s*)ax\.set_title\("Demo HPO Parallel Coordinate Plot.*?"(?:, fontsize=\d+)?\)'
        r'\s*\n(\s*)ax\.grid\(True, axis="y", linestyle="--", alpha=0\.25\); ax\.legend\(loc="upper right"(?:, fontsize=\d+)?\)',
        lambda m: (
            '        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=14, ha="right", fontsize=18)\n'
            '        ax.tick_params(axis="x", pad=10, labelsize=18)\n'
            '        ax.tick_params(axis="y", labelsize=18)\n'
            '        ax.set_ylim(-0.04, 1.04); ax.set_ylabel("Normalized parameter value", fontsize=18)\n'
            '        ax.set_title("Demo HPO Parallel Coordinate Plot \u2014 high-performance corridor", fontsize=18)\n'
            '        ax.grid(True, axis="y", linestyle="--", alpha=0.25); ax.legend(loc="upper right", fontsize=13)'
        ),
        src,
        "B4: demo axes+legend",
    )

    # ── B5: main plot_hpo_parallel_coordinates colorbar + axes + legend ──────
    src = re_sub_once(
        r'(    sm = plt\.cm\.ScalarMappable\(norm=norm, cmap=cmap\)\s*\n'
        r'    sm\.set_array\(\[\]\)\s*\n'
        r'    )(?:_?cbar = fig\.colorbar\(sm, ax=ax, pad=0\.02\)\s*\n'
        r'    (?:_?cbar|fig\.colorbar\(sm.*?\))\.set_label\(.*?\)(?:, fontsize=\d+)?\s*\n'
        r'(?:    _?cbar\.ax\.tick_params\(labelsize=\d+\)\s*\n)?'
        r'|fig\.colorbar\(sm, ax=ax, pad=0\.02\)\.set_label\(.*?\)\s*\n)'
        r'\s*\n'
        r'    ax\.set_xticks\(xs\)\s*\n'
        r'    ax\.set_xticklabels\(labels.*?fontsize=\d+.*?\)\s*\n'
        r'    ax\.tick_params\(axis="x".*?\)\s*\n'
        r'(?:    ax\.tick_params\(axis="y".*?\)\s*\n)?'
        r'    ax\.set_ylim.*?\n'
        r'    ax\.set_ylabel.*?(?:fontsize=\d+.*?)?\n'
        r'    ax\.set_title.*?(?:fontsize=\d+.*?)?\n'
        r'    ax\.grid.*?\n'
        r'\s*\n'
        r'    (leg = ax\.legend\(\s*\n'
        r'        loc="upper right",\s*\n'
        r'        fontsize=\d+,)',
        lambda m: (
            '    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)\n'
            '    sm.set_array([])\n'
            '    cbar = fig.colorbar(sm, ax=ax, pad=0.02)\n'
            '    cbar.set_label("Trial number" if color_is_trial else _hpo_metric_display_label(metric_name, args), fontsize=18)\n'
            '    cbar.ax.tick_params(labelsize=18)\n'
            '\n'
            '    ax.set_xticks(xs)\n'
            '    ax.set_xticklabels(labels, rotation=14, ha="right", fontsize=18)\n'
            '    ax.tick_params(axis="x", pad=10, labelsize=18)\n'
            '    ax.tick_params(axis="y", labelsize=18)\n'
            '    ax.set_ylim(-0.04, 1.04)\n'
            '    ax.set_ylabel("Normalized parameter value", fontsize=18)\n'
            '    ax.set_title("HPO Parallel Coordinate Plot \u2014 high-performance corridor", fontsize=18)\n'
            '    ax.grid(True, axis="y", linestyle="--", alpha=0.25)\n'
            '\n'
            '    leg = ax.legend(\n'
            '        loc="upper right",\n'
            '        fontsize=13,'
        ),
        src,
        "B5: main parallel_coordinates colorbar+axes+legend-open",
    )

    # ── B6: main legend body (handlelength / frame / linewidth) ─────────────
    src = re_sub_once(
        r'(    leg\.get_frame\(\)\.set_alpha\(1\.0\)\s*\n'
        r'    leg\.get_frame\(\)\.set_facecolor\("white"\)\s*\n'
        r'    leg\.get_frame\(\)\.set_edgecolor\("black"\)\s*\n'
        r'    leg\.get_frame\(\)\.set_linewidth\()(\d+(?:\.\d+)?)'
        r'(\)\s*\n'
        r'    plt\.setp\(leg\.get_lines\(\), linewidth=)(\d+(?:\.\d+)?)'
        r'(\)\s*\n'
        r'\s*\n'
        r'    fig\.tight_layout)',
        r'\g<1>1.5\g<3>5.0\g<5>',
        src,
        "B6: main legend frame sizes",
    )

    # ── B7: main legend handlelength/borderpad/labelspacing ─────────────────
    src = re_sub_once(
        r'(    leg = ax\.legend\(\s*\n'
        r'        loc="upper right",\s*\n'
        r'        fontsize=\d+,\s*\n'
        r'        frameon=True,\s*\n'
        r'        fancybox=False,\s*\n'
        r'        framealpha=1\.0,\s*\n'
        r'        facecolor="white",\s*\n'
        r'        edgecolor="black",\s*\n'
        r'        handlelength=)(\d+(?:\.\d+)?)'
        r'(,\s*\n'
        r'        borderpad=)(\d+(?:\.\d+)?)'
        r'(,\s*\n'
        r'        labelspacing=)(\d+(?:\.\d+)?)'
        r'(,\s*\n    \)\s*\n\s*\n    leg\.get_frame)',
        r'\g<1>3.4\g<3>0.9\g<5>0.8\g<7>',
        src,
        "B7: main legend handle/border/spacing",
    )

    # ── Apply exact-string patches (label changes) ───────────────────────────
    for old, new, name in patches:
        if old not in src:
            errors.append(f"  MISSING (exact): {name}")
        else:
            count = src.count(old)
            src = src.replace(old, new, 1)
            print(f"  OK (exact, {count}x found): {name}")

    if errors:
        print("\n[WARN] The following patches could NOT be applied:")
        for e in errors:
            print(e)
        print("\nThese may already be applied, or the file version differs.")
    else:
        print("\nAll patches applied successfully.")

    return src


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    if not os.path.exists(in_path):
        print(f"ERROR: input file not found: {in_path}")
        sys.exit(1)

    print(f"Reading  {in_path} ...")
    src = open(in_path, encoding="utf-8").read()

    print("Applying patches ...")
    patched = apply_patches(src)

    print(f"Writing  {out_path} ...")
    open(out_path, "w", encoding="utf-8").write(patched)
    print("Done.")


if __name__ == "__main__":
    main()