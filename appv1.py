"""LCA fiber comparison dashboard (Shiny + Plotly)."""

from __future__ import annotations
import base64
import io
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly

APP_ROOT = Path(__file__).resolve().parent
DATA_FILE = Path(os.getcwd()) / "Dataset" / "processed" / "processed-data.csv"

# ── App constants ────────────────────────────────────────────────────────────

CHARTS_PER_ROW = 2
PIE_HOLE = 0.4
BAR_COLORS = px.colors.qualitative.Set2
BAR_LABEL_MAX_LEN = 14
_GROUPED_BAR_IMPACT_LABEL_FONT_SIZE = 10
_GROUPED_BAR_IMPACT_LABEL_CHAR_PX = 5.8
_GROUPED_BAR_IMPACT_LABEL_LINE_PX = 13
_GROUPED_BAR_IMPACT_LABEL_MIN_MARGIN = 72
_GROUPED_BAR_IMPACT_PLOT_WIDTH_PX = 760
BAR_Y_LABEL_RAW = "Raw"
BAR_Y_LABEL_NORMALIZED = "Normalized"
BAR_EMPTY_SELECTION_MSG = (
    "Choose at least one fiber type and one environmental impact."
)
SECTION_IMPACT_LOCK_MSG = (
    "Only impacts in this section can be selected here. "
    "Go to the All Potential Impact Factors tab to add more."
)
MISSING_VALUE_LABEL = "N/A"
MISSING_VALUE_COLOR = "#c0392b"
HEATMAP_MISSING_CELL_FILL = "#ffffff"
HEATMAP_MISSING_LINE_COLOR = "#aaaaaa"
HEATMAP_DATA_COLORSCALE = [
    [0.0, "#eff3ff"],
    [0.25, "#bdd7e7"],
    [0.5, "#6baed6"],
    [0.75, "#3182bd"],
    [1.0, "#08519c"],
]
DATA_POLL_SECONDS = 10
SELECTION_DEBOUNCE_SECONDS = 0.4

NORM_METHOD_RAW = "raw"
NORM_METHOD_ABS_MAX = "abs_max"
NORM_METHOD_PER_IMPACT = NORM_METHOD_ABS_MAX
NORM_METHOD_CHOICES = {
    NORM_METHOD_RAW: "Raw data",
    NORM_METHOD_ABS_MAX: "Normalized data",
}


def _normalization_supported(
    fiber_count: int,
    *,
    method: str,
) -> bool:
    """Whether normalized values can be computed for the current fiber count."""
    if method == NORM_METHOD_RAW:
        return False
    return fiber_count >= 1


DISPLAY_DECIMALS = 3
DISPLAY_NUMBER_FMT = f".{DISPLAY_DECIMALS}f"
RADAR_RADIAL_TICK_FMT = ".0f"
RADAR_ABS_MAX_TICKVALS = [-1.0, -0.5, 0.0, 0.5, 1.0]
RADAR_ABS_MAX_TICKTEXT = ["-1", "-0.5", "0", "0.5", "1"]


def _bar_y_tickformat(norm_method: str) -> str:
    """Bar chart value-axis tick labels (less precision than table display)."""
    if norm_method == NORM_METHOD_ABS_MAX:
        return ".1f"
    return ".4g"


def _json_safe_number(value: object) -> float | None:
    """Plotly / Shiny JSON cannot serialize NaN or Inf."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _round_display_number(value: object) -> float | None:
    safe = _json_safe_number(value)
    if safe is None:
        return None
    return round(safe, DISPLAY_DECIMALS)


def _format_display_number(value: object, *, suffix: str = "") -> str:
    rounded = _round_display_number(value)
    if rounded is None:
        return MISSING_VALUE_LABEL
    if rounded == 1.0 or rounded == -1.0:
        return f"{int(rounded)}{suffix}"
    return f"{rounded:{DISPLAY_NUMBER_FMT}}{suffix}"


def _json_safe_numbers(values: pd.Series | list[object]) -> list[float | None]:
    if isinstance(values, pd.Series):
        values = values.tolist()
    return [_json_safe_number(v) for v in values]


def _polar_trace_coords(
    theta_labels: list[str],
    r_values: pd.Series,
    raw_values: pd.Series | None = None,
) -> tuple[list[str], list[float], list[float] | None]:
    """Drop non-finite points; Plotly polar maps null/NaN to stray dots near r=0."""
    thetas: list[str] = []
    rs: list[float] = []
    raw_out: list[float] = []
    has_raw = raw_values is not None
    for i, theta in enumerate(theta_labels):
        r = _json_safe_number(r_values.iloc[i])
        if r is None:
            continue
        thetas.append(theta)
        rs.append(r)
        if has_raw:
            raw_out.append(_json_safe_number(raw_values.iloc[i]) or 0.0)
    # Close the polygon for line-only traces (fill="toself" used to do this implicitly).
    if len(thetas) >= 3:
        thetas.append(thetas[0])
        rs.append(rs[0])
        if has_raw:
            raw_out.append(raw_out[0])
    return thetas, rs, (raw_out if has_raw else None)


def _sanitize_display_table(table: pd.DataFrame) -> pd.DataFrame:
    """Replace non-finite numeric cells with None for JSON-safe tables."""
    if table.empty:
        return table
    out = table.copy()
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].map(_json_safe_number)
    return out

PIE_SKIP_MSG = (
    "Pie chart not shown: this impact has negative or non-positive totals "
    "for the selected fibers. Use the Bar Chart tab to compare signed values."
)

_FORMULA_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
  stroke-linejoin="round" aria-hidden="true">
  <rect x="4" y="2" width="16" height="20" rx="2"/>
  <path d="M8 6h8M8 10h8M8 14h5"/>
  <path d="M9 18h6"/>
</svg>
"""

_TABLE_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
  stroke-linejoin="round" aria-hidden="true">
  <rect x="3" y="4" width="18" height="16" rx="2"/>
  <path d="M3 10h18M3 14h18M9 4v16M15 4v16"/>
</svg>
"""

_GUIDE_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
  fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
  stroke-linejoin="round" aria-hidden="true">
  <circle cx="12" cy="12" r="10"/>
  <path d="M12 11v5"/>
  <circle cx="12" cy="7.5" r="0.5" fill="currentColor"/>
</svg>
"""

# Hide Plotly modebar (zoom/pan/download icons) in the Shiny app.
PLOTLY_WIDGET_CONFIG = {"displayModeBar": False, "responsive": True}
CHART_EXPORT_SCALE = 4
CHART_EXPORT_MIN_WIDTH = 1680
CHART_EXPORT_MIN_HEIGHT = 960


@dataclass(frozen=True)
class ChartLayout:
    """Plotly layout knobs; spacing and title offset scale together."""

    row_px: int = 440
    legend_px: int = 56
    title_pad_px: int = 24
    single_height: int = 520
    bar_row_px: int = 400

    vertical_spacing: float = 0.05
    bar_vertical_spacing: float = 0.09
    horizontal_spacing: float = 0.06
    title_yshift_ratio: float = 10 / 1950

    margin_single: dict = field(
        default_factory=lambda: {"l": 30, "r": 30, "t": 60, "b": 40}
    )
    margin_grid: dict = field(
        default_factory=lambda: {"l": 20, "r": 20, "t": 28, "b": 28}
    )
    margin_bar_grid: dict = field(
        default_factory=lambda: {"l": 56, "r": 20, "t": 28, "b": 64}
    )

    def vertical_spacing_for(self, grid_rows: int, *, bar: bool = False) -> float:
        if grid_rows <= 1:
            return 0.0
        cap = 1 / (grid_rows - 1) - 0.01
        spacing = self.bar_vertical_spacing if bar else self.vertical_spacing
        return min(spacing, cap)

    def grid_height(self, row_count: int, *, bar: bool = False) -> int:
        px_per_row = self.bar_row_px if bar else self.row_px
        return row_count * px_per_row + self.legend_px + self.title_pad_px

    def title_yshift(self) -> int:
        return int(self.row_px * self.title_yshift_ratio)


LAYOUT = ChartLayout()


@dataclass
class LcaData:
    df: pd.DataFrame
    impact_col: str
    fiber_cols: list[str]
    impact_choices: list[str]
    mtime: float


def _normalize_catalog_label(text: str) -> str:
    """Trim and collapse whitespace; keep CSV spelling as-is."""
    return " ".join(str(text).strip().split())


def _clean_loaded_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and numeric fiber columns from the CSV."""
    out = raw.copy()
    out.columns = [_normalize_catalog_label(str(c).strip()) for c in out.columns]
    drop_cols = [
        c
        for c in out.columns
        if not c
        or str(c).startswith("Unnamed")
        or str(c).lower() == "nan"
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
    if out.empty:
        return out

    impact_col = out.columns[0]
    out = out[
        out[impact_col].notna()
        & (out[impact_col].astype(str).str.strip() != "")
    ].copy()
    out[impact_col] = (
        out[impact_col].astype(str).str.strip().map(_normalize_catalog_label)
    )

    non_fiber = {impact_col, "Units"}
    for col in out.columns:
        if col in non_fiber:
            continue
        out[col] = pd.to_numeric(
            out[col].astype(str).str.strip().replace({"*": "", "—": ""}),
            errors="coerce",
        )
    return out.reset_index(drop=True)


def load_processed_data(path: Path) -> LcaData:
    if not path.is_file():
        return LcaData(pd.DataFrame(), "", [], [], 0.0)

    try:
        # Attempt to read the file with utf-8 encoding
        raw = pd.read_csv(path, encoding="utf-8").dropna(how="all")
    except UnicodeDecodeError:
        # Fallback to iso-8859-1 encoding if utf-8 fails
        raw = pd.read_csv(path, encoding="iso-8859-1").dropna(how="all")

    if raw.empty:
        return LcaData(raw, "", [], [], path.stat().st_mtime)

    raw = _clean_loaded_dataframe(raw)
    if raw.empty:
        return LcaData(raw, "", [], [], path.stat().st_mtime)

    impact_col = raw.columns[0]
    non_fiber = {impact_col, "Units"}
    fiber_cols = [c for c in raw.columns if c not in non_fiber]
    impact_choices = raw[impact_col].astype(str).tolist()
    return LcaData(
        df=raw,
        impact_col=impact_col,
        fiber_cols=fiber_cols,
        impact_choices=impact_choices,
        mtime=path.stat().st_mtime,
    )


def _data_catalog_changed(previous: LcaData, current: LcaData) -> bool:
    if previous.mtime != current.mtime:
        return True
    if previous.fiber_cols != current.fiber_cols:
        return True
    if previous.impact_choices != current.impact_choices:
        return True
    if previous.df.shape != current.df.shape:
        return True
    return False


def _merge_selection(previous: list[str], choices: list[str]) -> list[str]:
    return [item for item in previous if item in choices]


def _selection_after_catalog_reload(
    current_selected: list[str],
    new_choices: list[str],
    *,
    all_was_selected: bool,
) -> list[str]:
    """Keep valid picks; new catalog items stay unselected unless All was on."""
    if all_was_selected:
        return list(new_choices)
    return [item for item in current_selected if item in new_choices]


INITIAL_DATA = load_processed_data(DATA_FILE)


@dataclass
class AppliedState:
    """Snapshot of sidebar selections; charts/tables read fibers and impacts."""

    fibers: list[str]
    impacts: list[str]


def _initial_applied_state() -> AppliedState:
    return AppliedState(fibers=[], impacts=[])


@dataclass(frozen=True)
class ChartKind:
    """Register chart types here; section tabs pick from this list automatically."""

    id: str
    label: str
    min_impacts: int = 1
    supports_normalization: bool = True
    uses_plotly: bool = True


CHART_KINDS: tuple[ChartKind, ...] = (
    ChartKind("dataframe", "DataFrame", uses_plotly=False),
    ChartKind("heatmap", "Heatmap", supports_normalization=False),
    ChartKind("bar", "Bar Chart"),
    ChartKind("bar_vertical", "Vertical Barchart"),
    ChartKind("bar_horizontal", "Horizontal Barchart"),
    ChartKind("radar", "Radar", min_impacts=3),
)

CHART_KIND_BY_ID: dict[str, ChartKind] = {k.id: k for k in CHART_KINDS}

IMPACT_SECTIONS: dict[str, dict[str, object]] = {
    "climate": {
        "tab": "Climate Impacts",
        "impacts": [
            "Climate change",
            "Climate change incl biogenic carbon",
        ],
    },
    "resources": {
        "tab": "Resources Consumption",
        "impacts": [
            "Freshwater consumption",
            "Land Use",
            "Fossil depletion",
        ],
    },
    "water": {
        "tab": "Ecosystem Impacts",
        "impacts": [
            "Freshwater ecotoxicity",
            "Freshwater eutrophication",
            "Acidification",
        ],
    },
    "toxicity_air": {
        "tab": "Human Health Impacts",
        "impacts": [
            "Human toxicity - Cancer",
            "Human toxicity - noncancer",
        ],
    },
}

IMPACT_DISPLAY_NAMES: dict[str, str] = {
    "climate change": "Climate Change",
    "climate change incl biogenic carbon": "Climate Change with Biogenic Carbon",
    "freshwater consumption": "Water Use",
    "land use": "Land Use",
    "fossil depletion": "Fossil Depletion",
    "freshwater ecotoxicity": "Freshwater Ecotoxicity",
    "freshwater eutrophication": "Freshwater Eutrophication",
    "acidification": "Acidification",
    "human toxicity - cancer": "Human Toxicity (Cancer)",
    "human toxicity - noncancer": "Human Toxicity (Non-Cancer)",
}

MAIN_TAB_TO_SECTION: dict[str, str] = {
    name: key
    for key, spec in IMPACT_SECTIONS.items()
    for name in (key, str(spec["tab"]))
}

TAB_HOME = "home"
TAB_COMPARISON = "comparison_table"
TAB_ALL_IMPACTS = "all_impacts"
TAB_METHOD = "method"
TAB_TEAM = "team_contributions"
TAB_REFERENCES = "references"
APP_TITLE = "Life Cycle Assessment of Select Textile Fibers"

_TEAM_MEMBER = dict[str, str | list[str]]
_TEAM_ORG_INFO = dict[str, str]
_TEAM_ORG_GROUP = dict[str, str | list[_TEAM_MEMBER]]
_TEAM_FUNCTION_AREA = dict[str, str | list[_TEAM_ORG_GROUP]]

TEAM_ORG_REGISTRY: dict[str, _TEAM_ORG_INFO] = {
    "nc_state": {
        "name": "North Carolina State University",
        "logo": "NC-State-Logo.png",
        "website": "https://www.ncsu.edu/",
    },
    "nlr": {
        "name": "National Laboratory of the Rockies",
        "logo": "NLR-logo.jpg",
        "website": "https://www.nlr.gov/",
    },
    "textile_engine": {
        "name": "Textile Innovation Engine",
        "logo": "Engine-logo.png",
        "website": "https://textileinnovationengine.org/",
    },
    "walmart": {
        "name": "Walmart Foundation",
        "logo": "walmart-logo.webp",
        "website": "https://www.walmart.org/",
    },
}

PARTNER_ORG_ORDER: tuple[str, ...] = (
    "nc_state",
    "textile_engine",
    "walmart",
    "nlr",
)

TEAM_FUNCTION_AREAS: tuple[_TEAM_FUNCTION_AREA, ...] = (
    {
        "title": "LCA Research",
        "orgs": [
            {
                "org_id": "nc_state",
                "members": [
                    {
                        "name": "Dr. Karen K. Leonas",
                        "role": "Lead PI and Project Lead",
                        "contributions": "LCA model development",
                    },
                    {
                        "name": "AYA Imam",
                        "role": "Graduate Student",
                        "contributions": "LCA model development",
                    },
                    {
                        "name": "Gloria Liu",
                        "role": "Graduate Student",
                        "contributions": "LCA model development",
                    },
                ],
            },
            {
                "org_id": "nlr",
                "members": [
                    {"name": "Katrina Kanuar"},
                    {"name": "Elisabeth Van Roijen"},
                    {"name": "Taylor Urkert"},
                ],
            },
        ],
    },
    {
        "title": "Data Visualization and Website Development",
        "orgs": [
            {
                "org_id": "nc_state",
                "members": [
                    {
                        "name": "Dr. Zahra Saki",
                        "role": "Co-PI",
                        "contributions": (
                            "Data visualization and website development"
                        ),
                    },
                    {
                        "name": "Ehsan Faghih",
                        "role": "Graduate Student",
                        "contributions": (
                            "Data visualization and website development"
                        ),
                    },
                ],
            },
        ],
    },
)

_HOME_SECTION_DESCRIPTIONS: dict[str, str] = {
    "climate": "Climate Change and biogenic carbon for selected fibers.",
    "resources": "Water Use, Land Use, and Fossil Depletion by fiber type.",
    "water": "Freshwater Ecotoxicity, Eutrophication, and Acidification.",
    "toxicity_air": "Human Toxicity (Cancer) and Human Toxicity (Non-Cancer).",
}

HOME_SECTION_CARDS: tuple[dict[str, str], ...] = tuple(
    {
        "button_id": f"home_nav_{section_key}",
        "tab": section_key,
        "title": str(IMPACT_SECTIONS[section_key]["tab"]),
        "description": _HOME_SECTION_DESCRIPTIONS[section_key],
    }
    for section_key in IMPACT_SECTIONS
)

NORM_METHOD_RADAR_ID = "norm_method_radar"
NORM_METHOD_BAR_ALL_ID = "norm_method_bar_all"


def _norm_radio_ids() -> tuple[str, ...]:
    return (
        NORM_METHOD_RADAR_ID,
        NORM_METHOD_BAR_ALL_ID,
        *(f"norm_method_{section_key}" for section_key in IMPACT_SECTIONS),
    )


# ── UI styles (shared gradients) ─────────────────────────────────────────────

_GRADIENT_TAB_DEFAULT = (
    "radial-gradient(ellipse 120% 100% at 50% 30%, "
    "#f8fbff 0%, #e7f1ff 38%, #cfe2ff 72%, #b8d4fe 100%)"
)
_GRADIENT_TAB_HOVER = (
    "radial-gradient(ellipse 120% 100% at 50% 28%, "
    "#ffffff 0%, #edf4ff 40%, #d7e9ff 100%)"
)
_GRADIENT_TAB_ACTIVE = (
    "radial-gradient(ellipse 120% 100% at 50% 28%, "
    "#4d9bff 0%, #0d6efd 38%, #0a58ca 72%, #084298 100%)"
)
_GRADIENT_TAB_ACTIVE_HOVER = (
    "radial-gradient(ellipse 120% 100% at 50% 26%, "
    "#5aa3ff 0%, #1a75ff 38%, #0b5ed7 72%, #084298 100%)"
)
_GRADIENT_HOME_CARD = (
    "radial-gradient(ellipse 120% 110% at 50% 18%, "
    "#ffffff 0%, #f8fbff 32%, #eef4ff 68%, #e3edff 100%)"
)
_GRADIENT_HOME_CARD_HOVER = (
    "radial-gradient(ellipse 120% 110% at 50% 15%, "
    "#ffffff 0%, #f3f8ff 30%, #e7f1ff 65%, #d7e9ff 100%)"
)
_GRADIENT_HOME_GROUP = (
    "radial-gradient(ellipse 130% 120% at 50% 12%, "
    "#f7faff 0%, #edf3ff 40%, #e2ebff 100%)"
)
_GRADIENT_HOME_SUBCARD = (
    "radial-gradient(ellipse 115% 105% at 50% 20%, "
    "#ffffff 0%, #fafcff 40%, #f0f6ff 100%)"
)
_GRADIENT_HOME_SUBCARD_HOVER = (
    "radial-gradient(ellipse 115% 105% at 50% 15%, "
    "#ffffff 0%, #f5f9ff 35%, #e8f2ff 100%)"
)
_GRADIENT_SIDEBAR_HEADER = (
    "radial-gradient(ellipse 120% 110% at 50% 18%, "
    "#ffffff 0%, #f8fbff 35%, #f0f6ff 72%, #e8f2ff 100%)"
)


def _section_tab_nav_css() -> str:
    """Fading blue pill background for all main nav tabs."""
    base_sel = "#main_tabs .nav-link"
    active_sel = "#main_tabs .nav-link.active"
    return f"""
{base_sel} {{
  background: {_GRADIENT_TAB_DEFAULT};
  border: 1px solid #b6d4fe;
  border-radius: 0.5rem 0.5rem 0 0;
  color: #0a58ca;
  font-weight: 600;
  padding: 0.45rem 0.95rem;
  margin-right: 0.25rem;
}}
{base_sel}:hover:not(.active) {{
  background: {_GRADIENT_TAB_HOVER};
  color: #084298;
  border-color: #9ec5fe;
}}
{active_sel} {{
  background: {_GRADIENT_TAB_ACTIVE};
  color: #ffffff;
  border-color: #084298;
  border-bottom-color: var(--bs-body-bg, #fff);
  box-shadow: 0 2px 6px rgba(8, 66, 152, 0.22);
}}
{active_sel}:hover,
{active_sel}:focus {{
  color: #ffffff;
  background: {_GRADIENT_TAB_ACTIVE_HOVER};
  border-color: #084298;
}}
"""


def _apply_button_css() -> str:
    """Apply button matches main nav tab pill styling."""
    return f"""
.lca-apply-btn.btn {{
  width: 100%;
  margin-top: 0.45rem;
  background: {_GRADIENT_TAB_DEFAULT};
  border: 1px solid #b6d4fe;
  border-radius: 0.5rem;
  color: #0a58ca;
  font-weight: 600;
  padding: 0.5rem 1rem;
}}
.lca-apply-btn.btn:hover,
.lca-apply-btn.btn:focus {{
  background: {_GRADIENT_TAB_HOVER};
  color: #084298;
  border-color: #9ec5fe;
  box-shadow: 0 0 0 0.12rem rgba(13, 110, 253, 0.2);
}}
.lca-applied-hidden {{
  display: none !important;
}}
"""


def _sidebar_note_css() -> str:
    return """
.lca-sidebar-note-wrap {
    margin-top: 0.65rem;
    padding: 0 0.35rem;
    text-align: center;
}
.lca-sidebar-note {
    font-size: 0.78rem;
    line-height: 1.5;
    color: #495057;
    margin: 0 auto;
    max-width: 18rem;
    text-align: center;
}
.lca-sidebar-note .lca-sidebar-method-link.action-link {
    display: inline;
    padding: 0;
    margin: 0;
    border: none;
    background: transparent;
    color: #0a58ca;
    font-size: inherit;
    font-weight: 600;
    text-decoration: underline;
    vertical-align: baseline;
    cursor: pointer;
}
.lca-sidebar-note .lca-sidebar-method-link.action-link:hover,
.lca-sidebar-note .lca-sidebar-method-link.action-link:focus {
    color: #084298;
    background: transparent;
    box-shadow: none;
}
"""


def _sidebar_methods_note_ui() -> ui.Tag:
    return ui.div(
        ui.tags.p(
            ui.tags.span(
                "To make validated consumer claims about these materials, you "
                "must follow the exact production, cultivation, and processing "
            ),
            ui.input_action_link(
                "nav_method_sidebar",
                "methods",
                class_="lca-sidebar-method-link",
            ),
            ui.tags.span(" described."),
            class_="lca-sidebar-note",
        ),
        class_="lca-sidebar-note-wrap",
    )


def _app_title_css() -> str:
    """Navbar title styled as a plain link-like control."""
    return """
.navbar .lca-app-title-btn.btn {
  color: #0a58ca !important;
  font-size: 1.25rem;
  font-weight: 600;
  line-height: 1.2;
  padding: 0;
  border: none;
  background: transparent !important;
  box-shadow: none !important;
  text-decoration: none;
  white-space: normal;
  text-align: left;
}
.navbar .lca-app-title-btn.btn:hover,
.navbar .lca-app-title-btn.btn:focus {
  color: #084298 !important;
  text-decoration: underline;
  background: transparent !important;
  box-shadow: none !important;
}
"""


def _partner_footer_css() -> str:
    return """
.lca-tab-page {
    display: flex;
    flex-direction: column;
    width: 100%;
    min-height: 100%;
    box-sizing: border-box;
}
.lca-tab-page-main {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
}
.lca-tab-page-main > .card {
    flex: 1 1 auto;
    min-height: 0;
}
.lca-tab-page--content .lca-tab-page-main {
    flex: 0 1 auto;
}
.lca-tab-page--content .lca-tab-page-main > .card {
    flex: 0 1 auto;
}
#main_tabs ~ .tab-content > .tab-pane.active {
    overflow-y: auto;
}
.lca-partner-footer {
    flex-shrink: 0;
    margin-top: 1.25rem;
    padding: 0.85rem 1.25rem 1rem;
    border-top: 1px solid #dee2e6;
    background: #f1f3f5;
}
.lca-partner-footer-logos {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 1.25rem 2rem;
}
.lca-partner-logo-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 5.5rem;
    min-height: 3rem;
    padding: 0.35rem 0.65rem;
    border-radius: 10px;
    background: transparent;
    text-decoration: none;
    cursor: pointer;
    filter: grayscale(100%);
    opacity: 0.68;
    transform: perspective(600px) translateY(0) scale(1);
    transition:
        filter 0.28s ease,
        opacity 0.28s ease,
        transform 0.28s ease,
        box-shadow 0.28s ease;
    box-shadow: 0 2px 4px rgba(15, 23, 42, 0);
}
.lca-partner-logo-link:hover,
.lca-partner-logo-link:focus {
    filter: grayscale(0%);
    opacity: 1;
    transform: perspective(600px) translateY(-4px) scale(1.07);
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.16);
    outline: none;
}
.lca-partner-logo-link:active {
    transform: perspective(600px) translateY(-1px) scale(1.03);
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.12);
}
.lca-partner-logo-img {
    display: block;
    max-height: 2.65rem;
    max-width: 8.5rem;
    width: auto;
    height: auto;
    object-fit: contain;
    pointer-events: none;
}
"""


def _plot_card_css() -> str:
    """Scrollable plot area when a chart card is expanded to full screen."""
    return """
.lca-chart-plot-wrap {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  min-height: 0;
  width: 100%;
}
.lca-chart-download-toolbar {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.35rem;
  padding: 0 0.25rem;
}
.lca-chart-plot-output {
  min-height: 0;
  width: 100%;
}
.bslib-card[data-full-screen="true"] {
  display: flex !important;
  flex-direction: column !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
.bslib-card[data-full-screen="true"] > .card-body {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: auto !important;
  max-height: none !important;
}
"""


def _chart_plot_output(output_id: str) -> ui.Tag:
    return ui.div(
        _chart_download_buttons(output_id),
        ui.div(output_widget(output_id), class_="lca-chart-plot-output"),
        class_="lca-chart-plot-wrap",
    )


def _all_impacts_chart_kind_choices() -> dict[str, str]:
    labels = {kind.id: kind.label for kind in CHART_KINDS}
    order = ("heatmap", "radar", "bar_vertical", "bar_horizontal")
    return {key: labels[key] for key in order}


def _impact_display_label(name: str) -> str:
    key = str(name).strip().lower()
    return IMPACT_DISPLAY_NAMES.get(key, _normalize_catalog_label(str(name)))


def _resolve_impacts_in_data(
    requested: list[str], available: list[str]
) -> list[str]:
    """Match section impact names to dataset rows (case-insensitive)."""
    by_key = {name.strip().lower(): name for name in available}
    matched: list[str] = []
    for name in requested:
        key = str(name).strip().lower()
        if key in by_key:
            matched.append(by_key[key])
    return matched


def _section_impact_names(section_key: str, available: list[str]) -> list[str]:
    spec = IMPACT_SECTIONS[section_key]
    return _resolve_impacts_in_data(list(spec["impacts"]), available)


def _section_impact_display_labels(
    section_key: str, available: list[str]
) -> list[str]:
    """User-facing labels for impacts present in the dataset for this section."""
    return [
        _impact_display_label(name)
        for name in _section_impact_names(section_key, available)
    ]


def _section_impacts_missing_from_data(
    section_key: str, available: list[str]
) -> list[str]:
    spec = IMPACT_SECTIONS[section_key]
    matched_lower = {
        name.strip().lower()
        for name in _section_impact_names(section_key, available)
    }
    return [
        _impact_display_label(str(name))
        for name in spec["impacts"]
        if str(name).strip().lower() not in matched_lower
    ]


def _impact_chip_bubbles_ui(
    names: list[str], *, muted: bool = False
) -> ui.Tag:
    if not names:
        return ui.tags.span("—", class_="text-muted")
    chip_class = (
        "lca-section-impact-chip lca-section-impact-chip-muted"
        if muted
        else "lca-section-impact-chip"
    )
    return ui.div(
        *[ui.tags.span(name, class_=chip_class) for name in names],
        class_="lca-section-impacts",
    )


def _impact_section_note_ui(section_key: str, available: list[str]) -> ui.Tag:
    display_labels = _section_impact_display_labels(section_key, available)
    missing = _section_impacts_missing_from_data(section_key, available)
    parts: list[ui.Tag] = [
        ui.div(
            ui.tags.p(
                "Impacts in this section:",
                class_="lca-formula-lead mb-2 fw-bold",
            ),
            _impact_chip_bubbles_ui(display_labels),
            class_="mb-2",
        ),
    ]
    if missing:
        parts.append(
            ui.div(
                ui.tags.p(
                    "Not found in the current dataset:",
                    class_="small text-muted mb-1",
                ),
                _impact_chip_bubbles_ui(missing, muted=True),
                class_="mb-2",
            )
        )
    parts.append(
        ui.tags.p(
            "Only impacts in this section are listed in the sidebar. "
            "You can select or unselect them, then click Apply. "
            "To compare other environmental impacts, open the "
            "All Potential Impact Factors tab.",
            class_="small text-muted mb-0",
        )
    )
    return ui.div(*parts, class_="px-3 pt-2")


def _impact_label(impact: str, row: pd.Series, include_units: bool) -> str:
    if not include_units or "Units" not in row.index:
        return str(impact)
    units = row["Units"]
    if pd.isna(units) or str(units).strip() == "":
        return str(impact)
    return f"{impact} ({units})"


def _row_for_impact(data: pd.DataFrame, impact_col: str, impact: str) -> pd.Series | None:
    match = data.loc[data[impact_col] == impact]
    if match.empty:
        return None
    return match.iloc[0]


def _fiber_values(row: pd.Series, fibers: list[str]) -> list[float | None]:
    return [
        _json_safe_number(row[f]) if f in row.index else None
        for f in fibers
    ]


def _pie_chart_block_reason(values: list[float | None]) -> str | None:
    numeric = [v for v in values if v is not None]
    if not numeric:
        return "No values to display (missing data for all selected fibers)."
    if all(v == 0 for v in numeric):
        return (
            "Pie chart not shown: all selected fiber values are zero. "
            "Use the Bar Chart tab."
        )
    if any(v < 0 for v in numeric):
        return PIE_SKIP_MSG
    if sum(numeric) <= 0:
        return PIE_SKIP_MSG
    return None


def _pie_labels_values(
    labels: list[str], values: list[float | None]
) -> tuple[list[str], list[float]]:
    """Drop missing fiber values before drawing a pie slice."""
    pairs = [(label, value) for label, value in zip(labels, values) if value is not None]
    if not pairs:
        return [], []
    out_labels, out_values = zip(*pairs)
    return list(out_labels), list(out_values)


def _format_missing_table_cells(
    table: pd.DataFrame,
    impact_col: str,
) -> pd.DataFrame:
    """Show N/A for empty / non-finite fiber cells in data tables."""
    if table.empty or "Message" in table.columns:
        return table
    out = table.copy()
    skip = {impact_col, "Units"}
    for col in out.columns:
        if col in skip:
            continue
        out[col] = out[col].map(
            lambda v: MISSING_VALUE_LABEL if _json_safe_number(v) is None else v
        )
    return out


def _subplot_titles(impacts: list[str], data: pd.DataFrame, impact_col: str) -> list[str]:
    titles: list[str] = []
    for impact in impacts:
        row = _row_for_impact(data, impact_col, impact)
        if row is None:
            titles.append(str(impact))
        else:
            titles.append(_impact_label(impact, row, include_units=True))
    return titles


def _grid_shape(n: int) -> tuple[int, int, int]:
    cols = CHARTS_PER_ROW
    grid_rows = (n + cols - 1) // cols
    pad = grid_rows * cols - n
    return cols, grid_rows, pad


def _bar_grid_shape(n: int) -> tuple[int, int, int]:
    """One full-width bar chart per impact."""
    return 1, n, 0


_ABS_MAX_TICK_TOLERANCE = 1e-4


def _finite_bar_values(values: list[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def _bar_has_abs_max_reference(values: list[float | None], target: float) -> bool:
    return any(
        np.isclose(v, target, atol=_ABS_MAX_TICK_TOLERANCE, rtol=0)
        for v in _finite_bar_values(values)
    )


def _linear_axis_tick_candidates(lo: float, hi: float, *, max_ticks: int = 6) -> list[float]:
    if hi <= lo:
        return [lo]
    step = (hi - lo) / max(max_ticks - 1, 1)
    if step <= 0:
        return [lo, hi]
    magnitude = 10 ** np.floor(np.log10(step))
    nice_step = magnitude * np.ceil(step / magnitude)
    start = np.floor(lo / nice_step) * nice_step
    ticks: list[float] = []
    tick = float(start)
    while tick <= hi + nice_step * 0.001:
        if lo - nice_step * 0.001 <= tick <= hi + nice_step * 0.001:
            ticks.append(tick)
        tick += float(nice_step)
    return ticks if ticks else [lo, hi]


def _bar_abs_max_axis_tickvals(
    lo: float,
    hi: float,
    values: list[float | None],
    *,
    max_ticks: int = 6,
) -> list[float]:
    """Per-impact norm: only show ±1 ticks when a bar actually has that value."""
    finite = _finite_bar_values(values)
    has_neg_one = _bar_has_abs_max_reference(values, -1.0)
    has_pos_one = _bar_has_abs_max_reference(values, 1.0)
    if finite:
        data_lo, data_hi = min(finite), max(finite)
        edge_pad = max((data_hi - data_lo) * 0.05, 0.01)
        tick_lo = data_lo - edge_pad
        tick_hi = data_hi + edge_pad
    else:
        tick_lo, tick_hi = lo, hi

    def _keep_tick(tick: float) -> bool:
        if np.isclose(tick, -1.0, atol=_ABS_MAX_TICK_TOLERANCE, rtol=0):
            return has_neg_one
        if np.isclose(tick, 1.0, atol=_ABS_MAX_TICK_TOLERANCE, rtol=0):
            return has_pos_one
        if np.isclose(tick, 0.0, atol=_ABS_MAX_TICK_TOLERANCE, rtol=0):
            return lo <= 0 <= hi
        return tick_lo - _ABS_MAX_TICK_TOLERANCE <= tick <= tick_hi + _ABS_MAX_TICK_TOLERANCE

    ticks = [tick for tick in _linear_axis_tick_candidates(lo, hi, max_ticks=max_ticks) if _keep_tick(tick)]
    if lo <= 0 <= hi and not any(
        np.isclose(t, 0.0, atol=_ABS_MAX_TICK_TOLERANCE, rtol=0) for t in ticks
    ):
        ticks.append(0.0)
    if has_neg_one and not any(
        np.isclose(t, -1.0, atol=_ABS_MAX_TICK_TOLERANCE, rtol=0) for t in ticks
    ):
        ticks.append(-1.0)
    if has_pos_one and not any(
        np.isclose(t, 1.0, atol=_ABS_MAX_TICK_TOLERANCE, rtol=0) for t in ticks
    ):
        ticks.append(1.0)
    ticks.sort()
    return ticks


def _bar_y_axis_range(
    values: list[float | None],
    *,
    norm_method: str = NORM_METHOD_RAW,
) -> list[float] | None:
    """Tight y-axis limits so bar heights reflect the actual value spread."""
    finite = _finite_bar_values(values)
    if not finite:
        return None

    mn, mx = min(finite), max(finite)
    if mn == mx:
        if mn == 0:
            return [-0.1, 0.1]
        pad = max(abs(mn) * 0.2, 0.05)
        return [mn - pad, mx + pad]

    span = mx - mn
    pad = max(span * 0.15, 0.03 * max(abs(mn), abs(mx), 1.0))

    if mn >= 0:
        lo = mn - pad
        hi = mx + pad
    elif mx <= 0:
        lo = mn - pad
        hi = mx + pad
    else:
        lo = min(mn - pad, -pad * 0.5)
        hi = max(mx + pad, pad * 0.5)

    return [lo, hi]


def _bar_axis_range_with_value_label_pad(
    axis_range: list[float],
    *,
    plot_px: int,
    value_label_px: int,
) -> list[float]:
    """Extend value-axis limits so outside bar value labels fit inside the subplot."""
    if plot_px <= 0 or value_label_px <= 0:
        return axis_range
    lo, hi = axis_range
    span = hi - lo
    if span <= 0:
        return axis_range
    pad = span * (value_label_px / plot_px)
    new_lo = lo - pad
    new_hi = hi + pad
    return [new_lo, new_hi]


def _apply_bar_y_axis_range(
    fig: go.Figure,
    values: list[float | None],
    *,
    norm_method: str,
    row: int | None = None,
    col: int | None = None,
    plot_px: int | None = None,
    value_label_px: int = 0,
    orientation: str = "v",
) -> None:
    axis_kw: dict = dict(row=row, col=col) if row is not None and col is not None else {}
    tickformat = _bar_y_tickformat(norm_method)
    axis_range = _bar_y_axis_range(values, norm_method=norm_method)
    if axis_range is None:
        axis_update = dict(tickformat=tickformat, **axis_kw)
        if orientation == "h":
            fig.update_xaxes(**axis_update)
        else:
            fig.update_yaxes(**axis_update)
        return
    if plot_px and value_label_px:
        axis_range = _bar_axis_range_with_value_label_pad(
            axis_range,
            plot_px=plot_px,
            value_label_px=value_label_px,
        )
    axis_update: dict = dict(
        range=axis_range,
        autorange=False,
        tickformat=tickformat,
        zeroline=axis_range[0] <= 0 <= axis_range[1],
        **axis_kw,
    )
    if norm_method == NORM_METHOD_ABS_MAX:
        lo, hi = axis_range
        axis_update["tickmode"] = "array"
        axis_update["tickvals"] = _bar_abs_max_axis_tickvals(lo, hi, values)
    if orientation == "h":
        fig.update_xaxes(**axis_update)
    else:
        fig.update_yaxes(**axis_update)


def _plotly_widget(fig: go.Figure) -> go.FigureWidget:
    """Return a FigureWidget with dashboard-friendly config (no modebar)."""
    widget = go.FigureWidget(fig)
    config = dict(PLOTLY_WIDGET_CONFIG)
    meta = fig.layout.meta
    if isinstance(meta, dict) and meta.get("responsive") is False:
        config["responsive"] = False
    widget._config = widget._config | config
    return widget


def _safe_download_basename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "-", str(name).strip().lower())
    return cleaned.strip("-") or "lca-chart"


def _figure_export_dimensions(fig: go.Figure) -> tuple[int, int]:
    width = fig.layout.width
    height = fig.layout.height
    if width is None:
        width = CHART_EXPORT_MIN_WIDTH
    if height is None:
        height = CHART_EXPORT_MIN_HEIGHT
    return max(int(width), CHART_EXPORT_MIN_WIDTH), max(
        int(height), CHART_EXPORT_MIN_HEIGHT
    )


def _export_plotly_image(fig: go.Figure, *, format: str) -> bytes:
    """High-resolution static export for PDF downloads."""
    width, height = _figure_export_dimensions(fig)
    export_fig = go.Figure(fig)
    export_fig.update_layout(width=width, height=height, autosize=False)
    buf = io.BytesIO()
    export_fig.write_image(
        buf,
        format=format,
        width=width,
        height=height,
        scale=CHART_EXPORT_SCALE,
        engine="kaleido",
    )
    return buf.getvalue()


def _chart_download_buttons(plot_output_id: str) -> ui.Tag:
    return ui.div(
        ui.download_button(
            f"{plot_output_id}_download_pdf",
            "Download",
            class_="btn btn-sm btn-outline-secondary",
        ),
        class_="lca-chart-download-toolbar",
    )


def _register_chart_download_handlers(
    output,
    render,
    *,
    plot_id: str,
    figure_fn: Callable[[], go.Figure],
    basename_fn: Callable[[], str],
) -> None:
    pdf_id = f"{plot_id}_download_pdf"

    def _pdf_handler():
        yield _export_plotly_image(figure_fn(), format="pdf")

    _pdf_handler.__name__ = pdf_id

    output(id=pdf_id)(
        render.download(
            filename=lambda: f"{_safe_download_basename(basename_fn())}.pdf",
            media_type="application/pdf",
        )(_pdf_handler)
    )


def _export_dataframe_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _table_download_buttons(table_output_id: str) -> ui.Tag:
    return ui.div(
        ui.download_button(
            f"{table_output_id}_download_csv",
            "Download",
            class_="btn btn-sm btn-outline-secondary",
        ),
        class_="lca-chart-download-toolbar",
    )


def _register_table_csv_download_handler(
    output,
    render,
    *,
    table_id: str,
    df_fn: Callable[[], pd.DataFrame],
    basename_fn: Callable[[], str],
) -> None:
    csv_id = f"{table_id}_download_csv"

    def _csv_handler():
        yield _export_dataframe_csv(df_fn())

    _csv_handler.__name__ = csv_id
    output(id=csv_id)(
        render.download(
            filename=lambda: f"{_safe_download_basename(basename_fn())}.csv",
            media_type="text/csv",
        )(_csv_handler)
    )


def _empty_figure(message: str, *, height: int | None = None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=message, x=0.5, xanchor="center"),
        height=height or LAYOUT.single_height,
        autosize=True,
        margin=LAYOUT.margin_single,
    )
    return fig


def _add_subplot_message(
    fig: go.Figure,
    message: str,
    *,
    row: int,
    col: int,
) -> None:
    """Show text in an xy subplot (domain/pie subplots cannot take annotations)."""
    fig.add_trace(
        go.Scatter(
            x=[0.5],
            y=[0.5],
            mode="text",
            text=[message],
            textposition="middle center",
            textfont=dict(size=11),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(
        range=[0, 1],
        visible=False,
        showticklabels=False,
        row=row,
        col=col,
    )
    fig.update_yaxes(
        range=[0, 1],
        visible=False,
        showticklabels=False,
        row=row,
        col=col,
    )


def _doughnut_grid_specs(
    impacts: list[str],
    pad: int,
    fibers: list[str],
    data: pd.DataFrame,
    impact_col: str,
) -> tuple[list[list[dict]], list[str | None]]:
    """Build mixed domain/xy specs and per-cell messages (None = draw pie)."""
    cells: list[str | None] = list(impacts) + [None] * pad
    specs_flat: list[dict] = []
    messages: list[str | None] = []

    for cell in cells:
        if cell is None:
            specs_flat.append({"type": "xy"})
            messages.append(None)
            continue

        row = _row_for_impact(data, impact_col, cell)
        if row is None:
            specs_flat.append({"type": "xy"})
            messages.append("No data for this impact.")
            continue

        reason = _pie_chart_block_reason(_fiber_values(row, fibers))
        if reason:
            specs_flat.append({"type": "xy"})
            messages.append(reason)
        else:
            specs_flat.append({"type": "domain"})
            messages.append(None)

    cols = CHARTS_PER_ROW
    grid_specs = [
        specs_flat[i : i + cols] for i in range(0, len(specs_flat), cols)
    ]
    return grid_specs, messages


def _yshift_subplot_titles(
    fig: go.Figure,
    layout: ChartLayout,
    *,
    yshift: int | None = None,
) -> None:
    """Shift subplot title annotations only (not fiber labels below bars)."""
    shift = yshift if yshift is not None else layout.title_yshift()
    for ann in fig.layout.annotations:
        yref = str(getattr(ann, "yref", "") or "")
        if "domain" in yref:
            continue
        ann.update(yshift=shift)


def _pie_trace(labels: list[str], values: list[float], *, showlegend: bool) -> go.Pie:
    return go.Pie(
        name="",
        labels=labels,
        values=values,
        hole=PIE_HOLE,
        textinfo="percent+label",
        textposition="inside",
        showlegend=showlegend,
        hovertemplate=(
            "<b>%{label}</b><br>"
            f"Value: %{{value:{DISPLAY_NUMBER_FMT}}}<br>"
            f"Share: %{{percent:{DISPLAY_NUMBER_FMT}}}<extra></extra>"
        ),
    )


def _bar_axis_labels(
    fibers: list[str],
    max_len: int = BAR_LABEL_MAX_LEN,
) -> tuple[list[str], list[str]]:
    """Full fiber names for data/hover; truncated names for axis display."""
    full = [str(f).strip() for f in fibers]
    display = [_truncate_display_label(label, max_len) for label in full]
    return full, display


def _bar_title_right_margin(titles: list[str]) -> int:
    """Room for vertical impact titles on the right edge of each chart."""
    max_len = max((len(title) for title in titles), default=40)
    return max(72, min(220, int(max_len * 5.2) + 24))


def _apply_bar_right_title(
    fig: go.Figure,
    title: str,
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Place the impact title vertically along the right side of the chart."""
    subplot_kw: dict = (
        dict(row=row, col=col) if row is not None and col is not None else {}
    )
    fig.add_annotation(
        text=title,
        x=1.03,
        y=0.5,
        xref="x domain",
        yref="y domain",
        textangle=-90,
        xanchor="left",
        yanchor="middle",
        showarrow=False,
        font=dict(size=11, color="#333"),
        **subplot_kw,
    )


def _bar_x_label_height_px(labels: list[str], *, max_len: int = BAR_LABEL_MAX_LEN) -> int:
    """Pixels needed below a subplot for vertical fiber labels."""
    longest = max(
        (min(len(str(label).strip()), max_len) for label in labels),
        default=max_len,
    )
    return max(100, int(longest * 7.5) + 40)


def _bar_y_label_width_px(labels: list[str], *, max_len: int = BAR_LABEL_MAX_LEN) -> int:
    """Pixels needed left of a subplot for horizontal category labels."""
    longest = max(
        (min(len(str(label).strip()), max_len) for label in labels),
        default=max_len,
    )
    return max(120, int(longest * 6.5) + 60)


def _bar_value_label_height_px() -> int:
    """Pixels needed above bar tops for outside value labels."""
    return 28


def _bar_subplot_gap_px(
    x_label_px: int,
    value_label_px: int,
    *,
    min_gap: int = 90,
    cap: int = 220,
) -> int:
    """Inter-row gap: room for upper x-labels or lower value labels, whichever is larger."""
    return min(max(x_label_px, value_label_px, min_gap), cap)


def _bar_chart_bottom_margin(labels: list[str], *, max_len: int = BAR_LABEL_MAX_LEN) -> int:
    """Room for vertical category labels below the last subplot."""
    return _bar_x_label_height_px(labels, max_len=max_len)


def _bar_grid_layout_dims(
    grid_rows: int,
    x_label_px: int,
    value_label_px: int,
    layout: ChartLayout = LAYOUT,
) -> dict[str, float | int]:
    """Vertical spacing and figure height for stacked full-width bar charts."""
    if grid_rows <= 1:
        return {
            "vertical_spacing": 0.0,
            "height": layout.single_height,
            "plot_px_per_row": max(300, layout.bar_row_px),
        }

    plot_px_per_row = max(300, layout.bar_row_px)
    label_gap_px = _bar_subplot_gap_px(x_label_px, value_label_px)

    top_margin = 60
    height = (
        grid_rows * plot_px_per_row
        + (grid_rows - 1) * label_gap_px
        + top_margin
        + x_label_px
        + 24
    )

    spacing = label_gap_px / height

    plotly_cap = 1 / (grid_rows - 1) - 0.01
    spacing = min(spacing, plotly_cap)

    return {
        "vertical_spacing": spacing,
        "height": height,
        "plot_px_per_row": plot_px_per_row,
    }


def _apply_bar_fiber_labels(
    fig: go.Figure,
    full_labels: list[str],
    display_labels: list[str],
    values: list[float | None],
    *,
    row: int | None = None,
    col: int | None = None,
    orientation: str = "v",
) -> None:
    """Category labels on the non-value axis; missing values use red text."""
    subplot_kw: dict = (
        dict(row=row, col=col) if row is not None and col is not None else {}
    )
    if orientation == "h":
        fig.update_yaxes(
            tickmode="array",
            tickvals=full_labels,
            ticktext=display_labels,
            showgrid=False,
            showline=False,
            tickfont=dict(size=10, color="#444"),
            **subplot_kw,
        )
        return

    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        showline=False,
        **subplot_kw,
    )
    for label, disp, value in zip(full_labels, display_labels, values):
        fig.add_annotation(
            x=label,
            y=-0.05,
            yref="y domain",
            text=disp,
            textangle=-90,
            xanchor="center",
            font=dict(
                size=10,
                color=MISSING_VALUE_COLOR if value is None else "#333333",
            ),
            showarrow=False,
            yanchor="top",
            **subplot_kw,
        )


def _grouped_bar_fiber_trace(
    *,
    name: str,
    x: list[str],
    y: list[float | None],
    color_idx: int,
    orientation: str = "v",
) -> go.Bar:
    """One trace per fiber; bars cluster at each impact category."""
    colors: list[str] = []
    texts: list[str] = []
    hover_values: list[str] = []
    safe_values: list[float | None] = []

    for value in y:
        safe = _json_safe_number(value)
        safe_values.append(safe)
        if safe is None:
            colors.append("rgba(210, 210, 210, 0.35)")
            texts.append(MISSING_VALUE_LABEL)
            hover_values.append(f"{MISSING_VALUE_LABEL} (missing)")
        else:
            colors.append(BAR_COLORS[color_idx % len(BAR_COLORS)])
            texts.append(_format_display_number(safe))
            hover_values.append(_format_display_number(safe))

    category_key = "%{y}" if orientation == "h" else "%{x}"
    bar_kw: dict = (
        dict(y=x, x=safe_values, orientation="h")
        if orientation == "h"
        else dict(x=x, y=safe_values)
    )

    return go.Bar(
        name=name,
        customdata=hover_values,
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.9)", width=1.5),
            opacity=0.92,
            cornerradius=6,
        ),
        text=texts,
        textposition="outside",
        textfont=dict(size=11, color="#444"),
        cliponaxis=False,
        hovertemplate=(
            f"<b>{category_key}</b><br>"
            "<b>%{fullData.name}</b><br>"
            "Value: %{customdata}<extra></extra>"
        ),
        **bar_kw,
    )


def _bar_impact_full_labels(
    impacts: list[str],
    data: pd.DataFrame,
    impact_col: str,
) -> list[str]:
    """Full impact names for grouped bar chart category positions."""
    full: list[str] = []
    for impact in impacts:
        row = _row_for_impact(data, impact_col, impact)
        if row is None:
            full.append(str(impact).strip())
        else:
            full.append(_impact_label(impact, row, include_units=False))
    return full


def _word_wrap_lines(text: str, max_chars: int) -> list[str]:
    """Wrap on word boundaries; single overlong tokens are hard-split."""
    text = str(text).strip()
    if not text:
        return []
    max_chars = max(4, max_chars)
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0

    def _flush() -> None:
        nonlocal current, current_len
        if current:
            lines.append(" ".join(current))
            current = []
            current_len = 0

    for word in words:
        if len(word) > max_chars:
            _flush()
            start = 0
            while start < len(word):
                lines.append(word[start : start + max_chars])
                start += max_chars
            continue
        extra = len(word) if not current else 1 + len(word)
        if current_len + extra <= max_chars:
            current.append(word)
            current_len += extra
        else:
            _flush()
            current = [word]
            current_len = len(word)
    _flush()
    return lines or [text[:max_chars]]


def _wrap_grouped_impact_label(
    text: str,
    *,
    max_chars_per_line: int,
    max_lines: int,
) -> str:
    """Wrap impact text into lines joined with <br> for Plotly labels."""
    text = str(text).strip()
    if not text:
        return ""
    max_lines = max(1, max_lines)
    max_chars = max(4, max_chars_per_line)
    for _ in range(48):
        lines = _word_wrap_lines(text, max_chars)
        if len(lines) <= max_lines:
            return "<br>".join(lines)
        max_chars += 2
    return "<br>".join(_word_wrap_lines(text, max_chars)[:max_lines])


def _grouped_bar_bundle_width_px(n_bundles: int, plot_width_px: int) -> int:
    if n_bundles <= 0:
        return plot_width_px
    return max(48, int(plot_width_px / n_bundles * 0.82))


def _grouped_bar_bundle_height_px(n_fibers: int) -> int:
    return max(44, 22 * n_fibers)


def _wrapped_label_vertical_extent_px(wrapped: str) -> int:
    """Pixels below the plot for textangle=-90 impact labels."""
    lines = wrapped.split("<br>") if wrapped else [""]
    longest = max((len(line) for line in lines), default=0)
    return int(longest * _GROUPED_BAR_IMPACT_LABEL_CHAR_PX + 16)


def _wrapped_label_horizontal_extent_px(wrapped: str) -> int:
    """Horizontal span of wrapped vertical impact labels (textangle=-90)."""
    lines = wrapped.split("<br>") if wrapped else [""]
    return int(len(lines) * _GROUPED_BAR_IMPACT_LABEL_LINE_PX + 8)


def _wrapped_label_left_extent_px(wrapped: str) -> int:
    """Left margin for horizontal bar multiline y-axis labels."""
    lines = wrapped.split("<br>") if wrapped else [""]
    longest = max((len(line) for line in lines), default=0)
    return int(longest * _GROUPED_BAR_IMPACT_LABEL_CHAR_PX + 16)


def _wrapped_label_row_height_px(wrapped: str) -> int:
    """Vertical space per impact row for horizontal bar labels."""
    lines = wrapped.split("<br>") if wrapped else [""]
    return int(len(lines) * _GROUPED_BAR_IMPACT_LABEL_LINE_PX + 8)


def _grouped_bar_impact_label_layout(
    full_labels: list[str],
    *,
    orientation: str,
    n_fibers: int,
    plot_width_px: int = _GROUPED_BAR_IMPACT_PLOT_WIDTH_PX,
) -> tuple[list[str], int]:
    """Wrap impact names within bundle bounds; return labels and margin px."""
    n = len(full_labels)
    if n == 0:
        return [], _GROUPED_BAR_IMPACT_LABEL_MIN_MARGIN

    if orientation == "h":
        bundle_px = _grouped_bar_bundle_height_px(n_fibers)
        max_lines = max(1, bundle_px // _GROUPED_BAR_IMPACT_LABEL_LINE_PX)
        margin_px = 132
        max_chars = max(12, int(margin_px / _GROUPED_BAR_IMPACT_LABEL_CHAR_PX))
        wrapped: list[str] = []
        for _ in range(3):
            wrapped = [
                _wrap_grouped_impact_label(
                    label,
                    max_chars_per_line=max_chars,
                    max_lines=max_lines,
                )
                for label in full_labels
            ]
            margin_px = max(
                _GROUPED_BAR_IMPACT_LABEL_MIN_MARGIN,
                max(_wrapped_label_left_extent_px(w) for w in wrapped) + 28,
            )
            max_chars = max(12, int(margin_px / _GROUPED_BAR_IMPACT_LABEL_CHAR_PX))
        return wrapped, margin_px

    bundle_px = _grouped_bar_bundle_width_px(n, plot_width_px)
    max_lines = max(1, bundle_px // _GROUPED_BAR_IMPACT_LABEL_LINE_PX)
    margin_px = 100
    max_chars = max(8, int(margin_px / _GROUPED_BAR_IMPACT_LABEL_CHAR_PX))
    wrapped = []
    for _ in range(3):
        max_lines = max(
            1,
            min(
                max_lines,
                bundle_px // _GROUPED_BAR_IMPACT_LABEL_LINE_PX,
            ),
        )
        wrapped = [
            _wrap_grouped_impact_label(
                label,
                max_chars_per_line=max_chars,
                max_lines=max_lines,
            )
            for label in full_labels
        ]
        too_wide = any(
            _wrapped_label_horizontal_extent_px(w) > bundle_px for w in wrapped
        )
        if too_wide and max_lines > 1:
            max_lines = max(1, max_lines - 1)
            continue
        margin_px = max(
            _GROUPED_BAR_IMPACT_LABEL_MIN_MARGIN,
            max(_wrapped_label_vertical_extent_px(w) for w in wrapped) + 28,
        )
        max_chars = max(8, int(margin_px / _GROUPED_BAR_IMPACT_LABEL_CHAR_PX))
    return wrapped, margin_px


def _apply_grouped_bar_impact_labels(
    fig: go.Figure,
    full_labels: list[str],
    display_labels: list[str],
    *,
    orientation: str = "v",
) -> None:
    """Multi-line impact labels that stay inside each bundle (between dotted lines)."""
    if orientation == "h":
        fig.update_yaxes(
            tickmode="array",
            tickvals=full_labels,
            ticktext=display_labels,
            showgrid=False,
            showline=False,
            tickfont=dict(
                size=_GROUPED_BAR_IMPACT_LABEL_FONT_SIZE,
                color="#444",
            ),
        )
        return

    fig.update_xaxes(
        tickmode="array",
        tickvals=full_labels,
        ticktext=[""] * len(full_labels),
        showticklabels=False,
        showgrid=False,
        showline=False,
    )
    for label, disp in zip(full_labels, display_labels):
        fig.add_annotation(
            x=label,
            y=-0.015,
            yref="y domain",
            text=disp,
            textangle=-90,
            xanchor="center",
            yanchor="top",
            showarrow=False,
            font=dict(
                size=_GROUPED_BAR_IMPACT_LABEL_FONT_SIZE,
                color="#333333",
            ),
        )


def _bar_fiber_legend() -> dict:
    return dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=10),
    )


def _apply_grouped_bar_bundle_separators(
    fig: go.Figure,
    n_bundles: int,
    *,
    orientation: str = "v",
) -> None:
    """Dotted lines between adjacent bar bundles."""
    if n_bundles < 2:
        return

    shapes = list(fig.layout.shapes or ())
    for i in range(n_bundles - 1):
        pos = i + 0.5
        if orientation == "h":
            shapes.append(
                dict(
                    type="line",
                    x0=0,
                    x1=1,
                    y0=pos,
                    y1=pos,
                    xref="paper",
                    yref="y",
                    line=dict(color="#c0c0c0", width=1, dash="dot"),
                    layer="below",
                )
            )
        else:
            shapes.append(
                dict(
                    type="line",
                    x0=pos,
                    x1=pos,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="y domain",
                    line=dict(color="#c0c0c0", width=1, dash="dot"),
                    layer="below",
                )
            )
    fig.update_layout(shapes=shapes)


def _bar_trace(
    labels: list[str],
    values: list[float | None],
    *,
    orientation: str = "v",
) -> go.Bar:
    colors: list[str] = []
    safe_values: list[float | None] = []
    text: list[str] = []
    hover_values: list[str] = []

    for i, value in enumerate(values):
        safe = _json_safe_number(value)
        safe_values.append(safe)
        if safe is None:
            colors.append("rgba(210, 210, 210, 0.35)")
            text.append(MISSING_VALUE_LABEL)
            hover_values.append(f"{MISSING_VALUE_LABEL} (missing)")
        else:
            colors.append(BAR_COLORS[i % len(BAR_COLORS)])
            text.append(_format_display_number(safe))
            hover_values.append(_format_display_number(safe))

    category_key = "%{y}" if orientation == "h" else "%{x}"
    bar_kw: dict = (
        dict(y=labels, x=safe_values, orientation="h")
        if orientation == "h"
        else dict(x=labels, y=safe_values)
    )

    return go.Bar(
        customdata=hover_values,
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.9)", width=1.5),
            opacity=0.92,
            cornerradius=6,
        ),
        text=text,
        textposition="outside",
        textfont=dict(size=11, color="#444"),
        cliponaxis=False,
        hovertemplate=(
            f"<b>{category_key}</b><br>"
            "Value: %{customdata}<extra></extra>"
        ),
        **bar_kw,
    )


def _apply_bar_chart_theme(
    fig: go.Figure,
    *,
    row: int | None = None,
    col: int | None = None,
    orientation: str = "v",
) -> None:
    """White plot background and light grid lines."""
    axis_kw: dict = dict(row=row, col=col) if row is not None and col is not None else {}

    if not axis_kw:
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")

    value_axis = dict(
        showline=True,
        linewidth=1,
        linecolor="#d0d0d0",
        mirror=False,
        showgrid=False,
        zeroline=True,
        zerolinecolor="#888888",
        zerolinewidth=1.5,
        tickfont=dict(size=10, color="#444"),
        title_font=dict(size=11, color="#333"),
        **axis_kw,
    )
    category_axis = dict(
        showline=False,
        mirror=False,
        showgrid=False,
        tickfont=dict(size=10, color="#444"),
        **axis_kw,
    )

    if orientation == "h":
        fig.update_xaxes(**value_axis)
        fig.update_yaxes(**category_axis)
    else:
        fig.update_xaxes(**category_axis)
        fig.update_yaxes(**value_axis)


def _legend_bottom(y: float = -0.08) -> dict:
    return dict(
        orientation="h",
        yanchor="bottom",
        y=y,
        xanchor="center",
        x=0.5,
    )


def _radar_figure_layout(
    n_legend_items: int,
    layout: ChartLayout = LAYOUT,
) -> tuple[int, dict, dict]:
    """Keep the polar plot full-sized; place many fiber labels in a side legend."""
    if n_legend_items <= 6:
        return (
            layout.single_height,
            dict(l=80, r=80, t=40, b=50),
            _legend_bottom(y=-0.08),
        )

    legend_margin_px = min(300, 120 + n_legend_items * 5)
    legend_content_px = n_legend_items * 15
    height = max(layout.single_height, legend_content_px + 120)
    margin = dict(l=80, r=legend_margin_px, t=40, b=40)
    legend = dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=0.99,
        font=dict(size=9),
        tracegroupgap=2,
        itemsizing="constant",
    )
    return height, margin, legend


# ── Figure builders ──────────────────────────────────────────────────────────

def build_doughnut_figure(
    impacts: list[str],
    fibers: list[str],
    data: pd.DataFrame,
    impact_col: str,
    *,
    layout: ChartLayout = LAYOUT,
) -> go.Figure:
    if not impacts or not fibers:
        return _empty_figure("Select at least one fiber and one impact.")

    labels = list(fibers)
    n = len(impacts)

    if n == 1:
        impact = impacts[0]
        row = _row_for_impact(data, impact_col, impact)
        if row is None:
            return _empty_figure("No data for this impact.")

        values = _fiber_values(row, fibers)
        reason = _pie_chart_block_reason(values)
        if reason:
            return _empty_figure(
                f"{_impact_label(impact, row, include_units=True)}\n\n{reason}",
            )

        pie_labels, pie_values = _pie_labels_values(labels, values)
        fig = go.Figure(
            data=[_pie_trace(labels=pie_labels, values=pie_values, showlegend=True)]
        )
        fig.update_layout(
            title=dict(
                text=_impact_label(impact, row, include_units=True),
                x=0.5,
                xanchor="center",
                pad=dict(t=8),
            ),
            height=layout.single_height,
            autosize=True,
            margin=layout.margin_single,
            legend=_legend_bottom(y=-0.12),
        )
        return fig

    cols, grid_rows, pad = _grid_shape(n)
    subplot_titles = _subplot_titles(impacts, data, impact_col)
    subplot_titles.extend([""] * pad)

    grid_specs, cell_messages = _doughnut_grid_specs(
        impacts, pad, fibers, data, impact_col
    )

    fig = make_subplots(
        rows=grid_rows,
        cols=cols,
        specs=grid_specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=layout.horizontal_spacing,
        vertical_spacing=layout.vertical_spacing_for(grid_rows),
    )

    cells: list[str | None] = list(impacts) + [None] * pad
    pie_legend_shown = False
    for idx, impact in enumerate(cells):
        r = idx // cols + 1
        c = idx % cols + 1
        message = cell_messages[idx]

        if message is not None:
            _add_subplot_message(fig, message, row=r, col=c)
            continue

        if impact is None:
            fig.update_xaxes(visible=False, row=r, col=c)
            fig.update_yaxes(visible=False, row=r, col=c)
            continue

        row = _row_for_impact(data, impact_col, impact)
        if row is None:
            continue

        pie_labels, pie_values = _pie_labels_values(labels, _fiber_values(row, fibers))
        if not pie_values:
            _add_subplot_message(
                fig,
                "No values to display (missing data for all selected fibers).",
                row=r,
                col=c,
            )
            continue

        fig.add_trace(
            _pie_trace(
                labels=pie_labels,
                values=pie_values,
                showlegend=not pie_legend_shown,
            ),
            row=r,
            col=c,
        )
        pie_legend_shown = True

    _yshift_subplot_titles(fig, layout)
    fig.update_layout(
        height=layout.grid_height(grid_rows),
        autosize=True,
        margin=layout.margin_grid,
        legend=_legend_bottom(),
    )
    return fig


@dataclass
class _BarNormContext:
    full_labels: list[str]
    display_labels: list[str]
    use_normalized: bool
    norm_note: str
    value_axis_label: str


def _bar_norm_context(
    impacts: list[str],
    fibers: list[str],
    data: pd.DataFrame,
    impact_col: str,
    norm_method: str,
) -> tuple[_BarNormContext, Callable[[str, pd.Series], list[float | None]]]:
    full_labels, display_labels = _bar_axis_labels(fibers)
    present_fibers = [f for f in fibers if f in data.columns]
    fiber_cols = [c for c in data.columns if c not in {impact_col, "Units"}]
    use_normalized = _normalization_supported(
        len(present_fibers),
        method=norm_method,
    )
    norm_matrix: pd.DataFrame | None = None
    if use_normalized:
        available = [i for i in impacts if i in set(data[impact_col].astype(str))]
        if available:
            matrix_fibers = _normalization_matrix_fibers(
                present_fibers,
                fiber_cols,
            )
            raw_matrix = _radar_value_matrix(
                data, impact_col, available, matrix_fibers
            )
            norm_matrix = _normalize_fiber_matrix(
                raw_matrix,
                present_fibers,
                method=norm_method,
                fiber_cols=fiber_cols,
            )

    def values_for_impact(impact: str, row: pd.Series) -> list[float | None]:
        if norm_matrix is not None and impact in norm_matrix.index:
            return [
                _json_safe_number(norm_matrix.at[impact, f])
                if f in norm_matrix.columns
                else None
                for f in full_labels
            ]
        return [_json_safe_number(v) for v in _fiber_values(row, fibers)]

    norm_note = ""
    value_axis_label = ""
    if use_normalized:
        norm_note, value_axis_label = _norm_chart_labels(norm_method)

    ctx = _BarNormContext(
        full_labels=full_labels,
        display_labels=display_labels,
        use_normalized=use_normalized,
        norm_note=norm_note,
        value_axis_label=value_axis_label,
    )
    return ctx, values_for_impact


def build_bar_figure(
    impacts: list[str],
    fibers: list[str],
    data: pd.DataFrame,
    impact_col: str,
    *,
    norm_method: str = NORM_METHOD_RAW,
    layout: ChartLayout = LAYOUT,
) -> go.Figure:
    if not impacts or not fibers:
        return _empty_figure(BAR_EMPTY_SELECTION_MSG)

    bar_ctx, values_for_impact = _bar_norm_context(
        impacts, fibers, data, impact_col, norm_method
    )
    full_labels = bar_ctx.full_labels
    display_labels = bar_ctx.display_labels
    use_normalized = bar_ctx.use_normalized
    norm_note = bar_ctx.norm_note
    norm_y_label = bar_ctx.value_axis_label
    x_label_px = _bar_x_label_height_px(full_labels)
    value_label_px = _bar_value_label_height_px()
    bottom_margin = x_label_px
    plot_px_per_row = max(300, layout.bar_row_px)
    n = len(impacts)

    if n == 1:
        impact = impacts[0]
        row = _row_for_impact(data, impact_col, impact)
        if row is None:
            return _empty_figure("No data for this impact.")

        values = values_for_impact(impact, row)
        fig = go.Figure(data=[_bar_trace(full_labels, values)])
        title = _impact_label(impact, row, include_units=True) + norm_note
        y_title = norm_y_label if use_normalized else BAR_Y_LABEL_RAW
        right_margin = _bar_title_right_margin([title])
        margin = {**layout.margin_single, "b": bottom_margin, "r": right_margin}
        fig.update_layout(
            height=layout.single_height,
            autosize=True,
            margin=margin,
            yaxis_title=y_title,
            bargap=0.35,
        )
        _apply_bar_chart_theme(fig)
        _apply_bar_y_axis_range(
            fig,
            values,
            norm_method=norm_method if use_normalized else NORM_METHOD_RAW,
            plot_px=plot_px_per_row,
            value_label_px=value_label_px,
        )
        _apply_bar_fiber_labels(fig, full_labels, display_labels, values)
        _apply_bar_right_title(fig, title)
        return fig

    cols, grid_rows, pad = _bar_grid_shape(n)
    impact_titles = _subplot_titles(impacts, data, impact_col)
    right_margin = _bar_title_right_margin(impact_titles)
    grid_dims = _bar_grid_layout_dims(grid_rows, x_label_px, value_label_px, layout)

    fig = make_subplots(
        rows=grid_rows,
        cols=cols,
        subplot_titles=[""] * (grid_rows * cols),
        vertical_spacing=grid_dims["vertical_spacing"],
        shared_xaxes=False,
        shared_yaxes=False,
    )

    for idx, impact in enumerate(impacts):
        row = _row_for_impact(data, impact_col, impact)
        r = idx // cols + 1
        c = idx % cols + 1
        if row is None:
            _add_subplot_message(fig, "No data for this impact.", row=r, col=c)
            continue

        values = values_for_impact(impact, row)
        fig.add_trace(_bar_trace(full_labels, values), row=r, col=c)
        y_label = norm_y_label if use_normalized else BAR_Y_LABEL_RAW
        fig.update_yaxes(title_text=y_label, row=r, col=c)
        _apply_bar_chart_theme(fig, row=r, col=c)
        _apply_bar_y_axis_range(
            fig,
            values,
            norm_method=norm_method if use_normalized else NORM_METHOD_RAW,
            row=r,
            col=c,
            plot_px=grid_dims["plot_px_per_row"],
            value_label_px=value_label_px,
        )
        _apply_bar_fiber_labels(
            fig, full_labels, display_labels, values, row=r, col=c
        )
        _apply_bar_right_title(
            fig,
            _impact_label(impact, row, include_units=True),
            row=r,
            col=c,
        )

    bar_grid_margin = {
        **layout.margin_bar_grid,
        "b": bottom_margin,
        "r": right_margin,
    }
    fig.update_layout(
        height=int(grid_dims["height"]),
        autosize=True,
        margin=bar_grid_margin,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        bargap=0.35,
    )
    return fig


def build_grouped_bar_figure(
    impacts: list[str],
    fibers: list[str],
    data: pd.DataFrame,
    impact_col: str,
    *,
    norm_method: str = NORM_METHOD_PER_IMPACT,
    layout: ChartLayout = LAYOUT,
    orientation: str = "v",
) -> go.Figure:
    """Grouped per-fiber bars (All Potential Impact Factors vertical/horizontal views)."""
    horizontal = orientation == "h"
    if not impacts or not fibers:
        return _empty_figure(BAR_EMPTY_SELECTION_MSG)

    bar_ctx, values_for_impact = _bar_norm_context(
        impacts, fibers, data, impact_col, norm_method
    )
    full_labels = bar_ctx.full_labels
    display_labels = bar_ctx.display_labels
    use_normalized = bar_ctx.use_normalized
    norm_note = bar_ctx.norm_note
    norm_value_label = bar_ctx.value_axis_label
    x_label_px = _bar_x_label_height_px(full_labels)
    y_label_px = _bar_y_label_width_px(full_labels)
    value_label_px = _bar_value_label_height_px()
    bottom_margin = x_label_px
    left_margin = y_label_px
    plot_px_per_row = max(300, layout.bar_row_px)
    n = len(impacts)

    if n == 1:
        impact = impacts[0]
        row = _row_for_impact(data, impact_col, impact)
        if row is None:
            return _empty_figure("No data for this impact.")

        values = values_for_impact(impact, row)
        title = _impact_label(impact, row, include_units=True) + norm_note
        value_title = norm_value_label if use_normalized else BAR_Y_LABEL_RAW

        if horizontal:
            fig = go.Figure(
                data=[_bar_trace(full_labels, values, orientation="h")]
            )
            height = max(layout.single_height, len(full_labels) * 40 + 100)
            margin = {
                **layout.margin_single,
                "l": left_margin,
                "b": 50,
                "r": 40,
            }
            fig.update_layout(
                height=height,
                autosize=True,
                margin=margin,
                xaxis_title=value_title,
                bargap=0.35,
                title=dict(text=title, x=0.5, xanchor="center", font=dict(size=12)),
            )
            _apply_bar_chart_theme(fig, orientation="h")
            _apply_bar_y_axis_range(
                fig,
                values,
                norm_method=norm_method if use_normalized else NORM_METHOD_RAW,
                plot_px=plot_px_per_row,
                value_label_px=value_label_px,
                orientation="h",
            )
            _apply_bar_fiber_labels(
                fig,
                full_labels,
                display_labels,
                values,
                orientation="h",
            )
            return fig

        fig = go.Figure(data=[_bar_trace(full_labels, values)])
        right_margin = _bar_title_right_margin([title])
        margin = {**layout.margin_single, "b": bottom_margin, "r": right_margin}
        fig.update_layout(
            height=layout.single_height,
            autosize=True,
            margin=margin,
            yaxis_title=value_title,
            bargap=0.35,
        )
        _apply_bar_chart_theme(fig)
        _apply_bar_y_axis_range(
            fig,
            values,
            norm_method=norm_method if use_normalized else NORM_METHOD_RAW,
            plot_px=plot_px_per_row,
            value_label_px=value_label_px,
        )
        _apply_bar_fiber_labels(fig, full_labels, display_labels, values)
        _apply_bar_right_title(fig, title)
        return fig

    impact_full = _bar_impact_full_labels(impacts, data, impact_col)
    impact_display, label_margin = _grouped_bar_impact_label_layout(
        impact_full,
        orientation=orientation,
        n_fibers=len(fibers),
    )
    if horizontal:
        bottom_margin = 50
        left_margin = label_margin
    else:
        bottom_margin = label_margin
        left_margin = layout.margin_single.get("l", 30)
    all_values: list[float | None] = []

    fig = go.Figure()
    for fiber_idx, fiber in enumerate(fibers):
        fiber_name = str(fiber).strip()
        y_vals: list[float | None] = []
        for impact in impacts:
            row = _row_for_impact(data, impact_col, impact)
            if row is None:
                y_vals.append(None)
                continue
            impact_values = values_for_impact(impact, row)
            y_vals.append(
                impact_values[fiber_idx] if fiber_idx < len(impact_values) else None
            )
        all_values.extend(y_vals)
        fig.add_trace(
            _grouped_bar_fiber_trace(
                name=fiber_name,
                x=impact_full,
                y=y_vals,
                color_idx=fiber_idx,
                orientation=orientation,
            )
        )

    value_title = norm_value_label if use_normalized else BAR_Y_LABEL_RAW
    if horizontal:
        row_px = max(
            _grouped_bar_bundle_height_px(len(fibers)),
            max(
                (_wrapped_label_row_height_px(w) for w in impact_display),
                default=44,
            ),
        )
        height = max(layout.single_height, len(impact_full) * row_px + 120)
        margin = {
            **layout.margin_single,
            "l": left_margin,
            "b": bottom_margin,
            "r": 30,
            "t": 72,
        }
        fig.update_layout(
            height=height,
            autosize=True,
            margin=margin,
            xaxis_title=value_title,
            barmode="group",
            bargap=0.22,
            bargroupgap=0.02,
            legend=_bar_fiber_legend(),
            showlegend=True,
        )
    else:
        margin = {**layout.margin_single, "b": bottom_margin, "r": 30, "t": 72}
        fig.update_layout(
            height=layout.single_height,
            autosize=True,
            margin=margin,
            yaxis_title=value_title,
            barmode="group",
            bargap=0.22,
            bargroupgap=0.02,
            legend=_bar_fiber_legend(),
            showlegend=True,
        )
    _apply_bar_chart_theme(fig, orientation=orientation)
    _apply_bar_y_axis_range(
        fig,
        all_values,
        norm_method=norm_method if use_normalized else NORM_METHOD_RAW,
        plot_px=plot_px_per_row,
        value_label_px=value_label_px,
        orientation=orientation,
    )
    _apply_grouped_bar_impact_labels(
        fig,
        impact_full,
        impact_display,
        orientation=orientation,
    )
    _apply_grouped_bar_bundle_separators(
        fig, len(impact_full), orientation=orientation
    )
    return fig


def _heatmap_row_color_scale(raw: pd.DataFrame, fibers: list[str]) -> pd.DataFrame:
    """Per-impact min–max to 0–1 for cell color when units differ across rows."""
    out = raw[fibers].copy()
    for impact in out.index:
        row = out.loc[impact, fibers]
        finite = row.dropna()
        if finite.empty:
            continue
        mn, mx = float(finite.min()), float(finite.max())
        if mx == mn:
            out.loc[impact, fibers] = 0.5
        else:
            out.loc[impact, fibers] = (row - mn) / (mx - mn)
    return out


def _truncate_display_label(text: str, max_len: int) -> str:
    """Shorten axis labels with trailing '...'; full text stays in hover."""
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return "..."
    return text[: max_len - 3] + "..."


def _heatmap_cell_dims(n_rows: int, n_cols: int) -> tuple[int, int]:
    """Pixel width/height per heatmap cell (narrow columns, taller rows)."""
    col_px = 22 if n_cols > 18 else (26 if n_cols > 14 else (30 if n_cols > 10 else (34 if n_cols > 7 else 38)))
    row_px = 44 if n_rows > 12 else (48 if n_rows > 8 else 52)
    return col_px, row_px


def _heatmap_impact_label_layout(
    full_labels: list[str],
    *,
    row_px: int,
) -> tuple[list[str], int]:
    """Wrap impact names for the y-axis within each row height."""
    if not full_labels:
        return [], _GROUPED_BAR_IMPACT_LABEL_MIN_MARGIN

    bundle_px = row_px
    max_lines = max(1, bundle_px // _GROUPED_BAR_IMPACT_LABEL_LINE_PX)
    margin_px = _GROUPED_BAR_IMPACT_LABEL_MIN_MARGIN
    max_chars = max(12, int(margin_px / _GROUPED_BAR_IMPACT_LABEL_CHAR_PX))
    wrapped: list[str] = []
    for _ in range(3):
        max_lines = max(
            1,
            min(max_lines, bundle_px // _GROUPED_BAR_IMPACT_LABEL_LINE_PX),
        )
        wrapped = [
            _wrap_grouped_impact_label(
                label,
                max_chars_per_line=max_chars,
                max_lines=max_lines,
            )
            for label in full_labels
        ]
        if any(_wrapped_label_row_height_px(w) > bundle_px for w in wrapped) and max_lines > 1:
            max_lines -= 1
            continue
        margin_px = max(
            _GROUPED_BAR_IMPACT_LABEL_MIN_MARGIN,
            max(_wrapped_label_left_extent_px(w) for w in wrapped) + 28,
        )
        max_chars = max(12, int(margin_px / _GROUPED_BAR_IMPACT_LABEL_CHAR_PX))
    return wrapped, margin_px


def _heatmap_fiber_label_layout(
    full_labels: list[str],
    *,
    col_px: int,
) -> tuple[list[str], int]:
    """Wrap fiber names for vertical x-axis labels within each column width."""
    if not full_labels:
        return [], 96

    bundle_px = col_px
    max_lines = max(1, bundle_px // _GROUPED_BAR_IMPACT_LABEL_LINE_PX)
    margin_px = 96
    max_chars = max(8, int(margin_px / _GROUPED_BAR_IMPACT_LABEL_CHAR_PX))
    wrapped: list[str] = []
    for _ in range(3):
        max_lines = max(
            1,
            min(max_lines, bundle_px // _GROUPED_BAR_IMPACT_LABEL_LINE_PX),
        )
        wrapped = [
            _wrap_grouped_impact_label(
                label,
                max_chars_per_line=max_chars,
                max_lines=max_lines,
            )
            for label in full_labels
        ]
        if any(
            _wrapped_label_horizontal_extent_px(w) > bundle_px for w in wrapped
        ) and max_lines > 1:
            max_lines -= 1
            continue
        margin_px = max(
            96,
            max(_wrapped_label_vertical_extent_px(w) for w in wrapped) + 28,
        )
        max_chars = max(8, int(margin_px / _GROUPED_BAR_IMPACT_LABEL_CHAR_PX))
    return wrapped, margin_px


def _heatmap_axis_labels(
    data: pd.DataFrame,
    impact_col: str,
    impacts: list[str],
    fibers: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Full labels for hover; wrapped labels for axis display."""
    y_full = _bar_impact_full_labels(impacts, data, impact_col)
    x_full = [str(f).strip() for f in fibers]
    col_px, row_px = _heatmap_cell_dims(len(y_full), len(x_full))
    y_display, _ = _heatmap_impact_label_layout(y_full, row_px=row_px)
    x_display, _ = _heatmap_fiber_label_layout(x_full, col_px=col_px)
    return y_full, y_display, x_full, x_display


def _heatmap_layout_dims(
    n_rows: int,
    n_cols: int,
    y_full: list[str],
    x_full: list[str],
    *,
    layout: ChartLayout,
) -> dict[str, object]:
    col_px, row_px = _heatmap_cell_dims(n_rows, n_cols)
    _, left = _heatmap_impact_label_layout(y_full, row_px=row_px)
    _, bottom = _heatmap_fiber_label_layout(x_full, col_px=col_px)
    top = 28
    heatmap_body_px = row_px * n_rows
    # Plot area height matches the heatmap grid; margins sit outside it.
    height = max(420, top + bottom + heatmap_body_px)
    plot_h = height - top - bottom
    dense = n_cols > 12 or n_rows * n_cols > 48
    return {
        "width": max(560, col_px * n_cols + 140 + 88),
        "height": height,
        "margin": dict(l=left, r=96, t=top, b=bottom),
        "tickfont_size": 8 if n_cols > 14 else (9 if n_cols > 10 else 10),
        "y_tickfont_size": _GROUPED_BAR_IMPACT_LABEL_FONT_SIZE,
        "dense": dense,
        "plot_h_px": plot_h,
        "n_rows": n_rows,
    }


def _apply_heatmap_fiber_labels(
    fig: go.Figure,
    full_labels: list[str],
    display_labels: list[str],
) -> None:
    """Multi-line vertical fiber labels below each heatmap column."""
    fig.update_xaxes(showticklabels=False)
    for label, disp in zip(full_labels, display_labels):
        fig.add_annotation(
            x=label,
            y=-0.02,
            yref="y domain",
            text=disp,
            textangle=-90,
            xanchor="center",
            yanchor="top",
            showarrow=False,
            font=dict(
                size=_GROUPED_BAR_IMPACT_LABEL_FONT_SIZE,
                color="#333333",
            ),
        )


def _heatmap_missing_cell_shapes(
    missing_cells: list[tuple[int, int]],
) -> list[dict]:
    """White tile with a diagonal line for missing heatmap cells."""
    shapes: list[dict] = []
    border = dict(color="#e0e0e0", width=1)
    slash = dict(color=HEATMAP_MISSING_LINE_COLOR, width=1)
    for col_idx, row_idx in missing_cells:
        x0, x1 = col_idx - 0.5, col_idx + 0.5
        y0, y1 = row_idx - 0.5, row_idx + 0.5
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                fillcolor=HEATMAP_MISSING_CELL_FILL,
                line=border,
                layer="above",
            )
        )
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=x0,
                y0=y1,
                x1=x1,
                y1=y0,
                line=slash,
                layer="above",
            )
        )
    return shapes


def _heatmap_colorbar() -> dict:
    """Vertical scale beside the heatmap grid (full plot height)."""
    return dict(
        orientation="v",
        thickness=22,
        len=1.0,
        lenmode="fraction",
        y=0.5,
        yanchor="middle",
        x=1.02,
        xpad=6,
        tickmode="array",
        tickvals=[0.0, 1.0],
        ticktext=["Min", "Max"],
        tickfont=dict(size=11, color="#333"),
        outlinewidth=1,
        outlinecolor="#cccccc",
        showticklabels=True,
    )


def build_heatmap_figure(
    impacts: list[str],
    fibers: list[str],
    data: pd.DataFrame,
    impact_col: str,
    *,
    layout: ChartLayout = LAYOUT,
) -> go.Figure:
    if not impacts or not fibers:
        return _empty_figure("Select at least one fiber and one impact.")

    present_fibers = [f for f in fibers if f in data.columns]
    available_impacts = _ordered_selected_impacts(data, impact_col, impacts)
    if not available_impacts or not present_fibers:
        return _empty_figure("No data for the selected fibers and impacts.")

    raw_matrix = _radar_value_matrix(
        data, impact_col, available_impacts, present_fibers
    )
    value_matrix = raw_matrix[present_fibers]

    # Each impact row is colored on its own min–max scale (impacts use different units).
    color_matrix = _heatmap_row_color_scale(value_matrix, present_fibers)

    y_full, y_display, x_full, x_display = _heatmap_axis_labels(
        data, impact_col, available_impacts, present_fibers
    )

    z: list[list[float | None]] = []
    hover_data: list[list[list[str]]] = []
    missing_cells: list[tuple[int, int]] = []
    for i, impact in enumerate(available_impacts):
        z_row: list[float | None] = []
        hover_row: list[list[str]] = []
        impact_full = y_full[i]
        for j, fiber in enumerate(present_fibers):
            display_val = _json_safe_number(value_matrix.at[impact, fiber])
            if display_val is None:
                z_row.append(None)
                missing_cells.append((j, i))
                value_text = f"{MISSING_VALUE_LABEL} (missing)"
            else:
                color_val = _json_safe_number(color_matrix.at[impact, fiber])
                if color_val is None:
                    z_row.append(None)
                    missing_cells.append((j, i))
                    value_text = f"{MISSING_VALUE_LABEL} (missing)"
                else:
                    z_row.append(color_val)
                    value_text = _format_display_number(display_val)
            hover_row.append([impact_full, fiber, value_text])
        z.append(z_row)
        hover_data.append(hover_row)

    dims = _heatmap_layout_dims(
        len(y_full),
        len(present_fibers),
        y_full,
        x_full,
        layout=layout,
    )

    heatmap_kw: dict = dict(
        z=z,
        x=x_full,
        y=y_full,
        zmin=0.0,
        zmax=1.0,
        colorscale=HEATMAP_DATA_COLORSCALE,
        showscale=True,
        colorbar=_heatmap_colorbar(),
        hoverongaps=True,
        customdata=hover_data,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Fiber: %{customdata[1]}<br>"
            "Value: %{customdata[2]}<extra></extra>"
        ),
        xgap=1,
        ygap=1,
    )

    n_rows = int(dims["n_rows"])
    fig = go.Figure(data=[go.Heatmap(**heatmap_kw)])
    fig.update_layout(
        width=dims["width"],
        height=dims["height"],
        autosize=not dims["dense"],
        margin=dims["margin"],
        paper_bgcolor="white",
        plot_bgcolor="white",
        shapes=_heatmap_missing_cell_shapes(missing_cells),
        xaxis=dict(
            tickmode="array",
            tickvals=x_full,
            ticktext=[""] * len(x_full),
            side="bottom",
            automargin=True,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_full,
            ticktext=y_display,
            range=[-0.5, n_rows - 0.5],
            autorange="reversed",
            tickfont=dict(size=dims["y_tickfont_size"], color="#444"),
        ),
        meta=dict(responsive=not dims["dense"]),
    )
    _apply_heatmap_fiber_labels(fig, x_full, x_display)
    return fig


def _short_theta_label(text: str, max_len: int = 36) -> str:
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _radar_value_matrix(
    data: pd.DataFrame,
    impact_col: str,
    impacts: list[str],
    fibers: list[str],
) -> pd.DataFrame:
    subset = data.loc[data[impact_col].isin(impacts)].set_index(impact_col)
    return subset.loc[[i for i in impacts if i in subset.index], fibers].apply(
        pd.to_numeric, errors="coerce"
    ).astype(float)


def _normalization_matrix_fibers(
    present_fibers: list[str],
    fiber_cols: list[str],
) -> list[str]:
    """Columns to load for normalization; one fiber uses full catalog for scaling."""
    if len(present_fibers) < 2:
        return list(fiber_cols)
    return list(present_fibers)


def _normalize_category_across_fibers_abs_max(
    raw: pd.DataFrame,
    fibers: list[str],
    *,
    scale_fibers: list[str] | None = None,
) -> pd.DataFrame:
    """Per impact: value / max(|value|) across scale_fibers → [-1, 1]."""
    present = [f for f in fibers if f in raw.columns]
    if not present:
        return raw[present].copy()

    scale = scale_fibers if scale_fibers is not None else present
    scale = [f for f in scale if f in raw.columns]
    if not scale:
        scale = present

    out = raw[present].copy()
    for impact in out.index:
        scale_row = raw.loc[impact, scale]
        max_abs = float(scale_row.abs().max())
        if pd.isna(max_abs) or max_abs == 0:
            out.loc[impact, present] = 0.0
        else:
            values = raw.loc[impact, present]
            out.loc[impact, present] = values / max_abs
    return out


def _range_normalization_scale_fibers(
    display_fibers: list[str],
    fiber_cols: list[str],
    raw: pd.DataFrame,
) -> list[str]:
    """Use selected fibers for scaling when 2+; otherwise scale against the full catalog."""
    present = [f for f in display_fibers if f in raw.columns]
    if len(present) >= 2:
        return present
    catalog = [f for f in fiber_cols if f in raw.columns]
    return catalog if catalog else present


def _normalize_fiber_matrix(
    raw: pd.DataFrame,
    fibers: list[str],
    *,
    method: str,
    fiber_cols: list[str] | None = None,
) -> pd.DataFrame:
    if method == NORM_METHOD_RAW:
        return raw.copy()
    catalog = fiber_cols if fiber_cols is not None else list(raw.columns)
    scale_fibers = _range_normalization_scale_fibers(fibers, catalog, raw)
    return _normalize_category_across_fibers_abs_max(
        raw, fibers, scale_fibers=scale_fibers
    )


def _coerce_norm_method(selected: list[str]) -> str:
    for method in selected:
        if method in NORM_METHOD_CHOICES:
            return method
    return NORM_METHOD_ABS_MAX


def _norm_table_decimals(method: str) -> int:
    return DISPLAY_DECIMALS


def _norm_chart_labels(method: str) -> tuple[str, str]:
    """Return (title_suffix, value_axis_label) for bar charts."""
    if method == NORM_METHOD_RAW:
        return ("", BAR_Y_LABEL_RAW)
    return ("", BAR_Y_LABEL_NORMALIZED)


def _build_normalized_data_table(
    data: pd.DataFrame,
    impact_col: str,
    impacts: list[str],
    fibers: list[str],
    *,
    method: str,
    fiber_cols: list[str] | None = None,
) -> pd.DataFrame:
    catalog_fibers = fiber_cols if fiber_cols is not None else [
        c for c in data.columns if c not in {impact_col, "Units"}
    ]
    available = _ordered_selected_impacts(data, impact_col, impacts)
    present_fibers = _ordered_selected_fibers(catalog_fibers, fibers)
    if not available or not present_fibers:
        return pd.DataFrame()

    cols = [impact_col]
    if "Units" in data.columns:
        cols.append("Units")
    cols.extend(present_fibers)

    table = data.loc[data[impact_col].isin(available), cols].copy()
    order_key = {name: idx for idx, name in enumerate(available)}
    table["_sort"] = table[impact_col].astype(str).map(order_key)
    table = table.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    matrix_fibers = _normalization_matrix_fibers(
        present_fibers,
        catalog_fibers,
    )
    raw_matrix = _radar_value_matrix(data, impact_col, available, matrix_fibers)
    norm_values = _normalize_fiber_matrix(
        raw_matrix,
        present_fibers,
        method=method,
        fiber_cols=catalog_fibers,
    )

    for fiber in present_fibers:
        table[fiber] = table[impact_col].map(
            lambda impact: norm_values.at[impact, fiber]
            if impact in norm_values.index
            else float("nan")
        )
    return _format_missing_table_cells(
        _sanitize_display_table(table.round(_norm_table_decimals(method))),
        impact_col,
    )


def _build_raw_radar_preview_table(
    data: pd.DataFrame,
    impact_col: str,
    impacts: list[str],
    fibers: list[str],
) -> pd.DataFrame:
    available = [i for i in impacts if i in set(data[impact_col].astype(str))]
    present_fibers = [f for f in fibers if f in data.columns]
    if not available or not present_fibers:
        return pd.DataFrame()

    cols = [impact_col]
    if "Units" in data.columns:
        cols.append("Units")
    cols.extend(present_fibers)
    return data.loc[data[impact_col].isin(available), cols].copy()


def _polar_axis_range(values: pd.Series) -> list[float]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return [0.0, 1.0]
    min_val = float(clean.min())
    max_val = float(clean.max())
    if min_val == max_val:
        pad = abs(max_val) * 0.15 if max_val != 0 else 1.0
        return [min_val - pad, max_val + pad]
    pad = (max_val - min_val) * 0.1
    return [min_val - pad, max_val + pad]


def build_radar_figure(
    impacts: list[str],
    fibers: list[str],
    data: pd.DataFrame,
    impact_col: str,
    *,
    norm_method: str = NORM_METHOD_RAW,
    layout: ChartLayout = LAYOUT,
) -> go.Figure:
    if not impacts or not fibers:
        return _empty_figure("Select at least one fiber and one environmental impact.")

    if data.empty or not impact_col:
        return _empty_figure("No data loaded.")

    if len(impacts) < 3:
        return _empty_figure(
            "Select at least 3 environmental impacts to draw a radar chart."
        )

    available = [i for i in impacts if i in set(data[impact_col].astype(str))]
    if len(available) < 3:
        return _empty_figure("Fewer than 3 selected impacts exist in the data file.")

    present_fibers = [f for f in fibers if f in data.columns]
    if not present_fibers:
        return _empty_figure("No fiber values found for the current selection.")

    fiber_cols = [c for c in data.columns if c not in {impact_col, "Units"}]
    matrix_fibers = _normalization_matrix_fibers(present_fibers, fiber_cols)
    raw = _radar_value_matrix(data, impact_col, available, matrix_fibers)
    raw_display = raw[present_fibers] if present_fibers else raw
    theta_labels = [_short_theta_label(i) for i in available]
    single_fiber = len(present_fibers) == 1

    if norm_method == NORM_METHOD_RAW:
        plot_values = raw_display
        radialaxis = dict(
            visible=True,
            range=_polar_axis_range(
                plot_values.iloc[:, 0] if single_fiber else plot_values.stack()
            ),
            tickformat=RADAR_RADIAL_TICK_FMT,
            gridcolor="#e0e0e0",
            linecolor="#cccccc",
        )
    else:
        plot_values = _normalize_fiber_matrix(
            raw,
            present_fibers,
            method=NORM_METHOD_ABS_MAX,
            fiber_cols=fiber_cols,
        )
        radialaxis = dict(
            visible=True,
            range=[-1.0, 1.0],
            tickmode="array",
            tickvals=RADAR_ABS_MAX_TICKVALS,
            ticktext=RADAR_ABS_MAX_TICKTEXT,
            gridcolor="#e0e0e0",
            linecolor="#cccccc",
        )

    fig = go.Figure()
    for i, fiber in enumerate(present_fibers):
        color = BAR_COLORS[i % len(BAR_COLORS)]
        trace_theta, trace_r, trace_raw_list = _polar_trace_coords(
            theta_labels,
            plot_values[fiber],
            raw_display[fiber] if norm_method != NORM_METHOD_RAW else None,
        )
        trace_custom = trace_raw_list
        if norm_method == NORM_METHOD_RAW:
            hovertemplate = (
                "<b>%{theta}</b><br>"
                f"{fiber}<br>"
                f"Value: %{{r:{DISPLAY_NUMBER_FMT}}}<extra></extra>"
            )
        else:
            hovertemplate = (
                "<b>%{theta}</b><br>"
                f"{fiber}<br>"
                f"Raw: %{{customdata:{DISPLAY_NUMBER_FMT}}}<br>"
                f"Normalized: %{{r:{DISPLAY_NUMBER_FMT}}}<extra></extra>"
            )
        if len(trace_r) < 3:
            continue
        polar_kw: dict = dict(
            r=trace_r,
            theta=trace_theta,
            name=fiber,
            customdata=trace_custom,
            line=dict(color=color, width=2),
            hovertemplate=hovertemplate,
        )
        if single_fiber and norm_method == NORM_METHOD_RAW:
            polar_kw["mode"] = "lines"
        else:
            polar_kw.update(
                mode="lines+markers",
                marker=dict(size=8, color=color, line=dict(width=1, color="white")),
                hoveron="points",
            )
        fig.add_trace(go.Scatterpolar(**polar_kw))

    if not fig.data:
        return _empty_figure("No fiber values found for the current selection.")

    height, margin, legend = _radar_figure_layout(
        len(present_fibers),
        layout=layout,
    )
    fig.update_layout(
        title=None,
        height=height,
        autosize=True,
        margin=margin,
        paper_bgcolor="white",
        plot_bgcolor="white",
        polar=dict(
            bgcolor="white",
            radialaxis=radialaxis,
            angularaxis=dict(
                direction="clockwise",
                gridcolor="#e0e0e0",
                linecolor="#cccccc",
            ),
        ),
        legend=legend,
        showlegend=not single_fiber,
        hovermode="closest",
    )
    return fig


def _build_chart_figure(
    chart_id: str,
    impacts: list[str],
    fibers: list[str],
    data: pd.DataFrame,
    impact_col: str,
    *,
    norm_method: str = NORM_METHOD_RAW,
) -> go.Figure:
    if chart_id == "heatmap":
        return build_heatmap_figure(impacts, fibers, data, impact_col)
    if chart_id == "bar_vertical":
        return build_grouped_bar_figure(
            impacts,
            fibers,
            data,
            impact_col,
            norm_method=norm_method,
            orientation="v",
        )
    if chart_id == "bar_horizontal":
        return build_grouped_bar_figure(
            impacts,
            fibers,
            data,
            impact_col,
            norm_method=norm_method,
            orientation="h",
        )
    if chart_id == "bar":
        return build_bar_figure(
            impacts,
            fibers,
            data,
            impact_col,
            norm_method=norm_method,
        )
    if chart_id == "radar":
        return build_radar_figure(
            impacts,
            fibers,
            data,
            impact_col,
            norm_method=norm_method,
        )
    return _empty_figure("Unknown chart type.")


def _ordered_selected_impacts(
    data: pd.DataFrame, impact_col: str, impacts: list[str]
) -> list[str]:
    """Keep CSV row order for the selected impact categories."""
    selected = set(impacts)
    return [
        str(name)
        for name in data[impact_col].astype(str)
        if str(name) in selected
    ]


def _ordered_selected_fibers(
    fiber_cols: list[str], fibers: list[str]
) -> list[str]:
    """Keep CSV column order for the selected fiber types."""
    selected = set(fibers)
    return [name for name in fiber_cols if name in selected]


def _build_raw_comparison_table(
    data: pd.DataFrame,
    impact_col: str,
    impacts: list[str],
    fibers: list[str],
    *,
    fiber_cols: list[str],
) -> pd.DataFrame:
    available_impacts = _ordered_selected_impacts(data, impact_col, impacts)
    present_fibers = _ordered_selected_fibers(fiber_cols, fibers)
    if not available_impacts or not present_fibers:
        return pd.DataFrame()

    cols = [impact_col]
    if "Units" in data.columns:
        cols.append("Units")
    cols.extend(present_fibers)
    table = data.loc[data[impact_col].isin(available_impacts), cols].copy()
    order_key = {name: idx for idx, name in enumerate(available_impacts)}
    table["_sort"] = table[impact_col].astype(str).map(order_key)
    table = table.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return _format_missing_table_cells(_sanitize_display_table(table), impact_col)


def _build_chart_data_table(
    impacts: list[str],
    fibers: list[str],
    data: pd.DataFrame,
    impact_col: str,
    *,
    norm_method: str,
    fiber_cols: list[str] | None = None,
) -> pd.DataFrame:
    if not fibers or not impacts:
        return pd.DataFrame({"Message": ["Select at least one fiber and one impact."]})
    catalog_fibers = fiber_cols if fiber_cols is not None else [
        c for c in data.columns if c not in {impact_col, "Units"}
    ]
    if norm_method == NORM_METHOD_RAW:
        return _build_raw_comparison_table(
            data,
            impact_col,
            impacts,
            fibers,
            fiber_cols=catalog_fibers,
        )
    return _sanitize_display_table(
        _build_normalized_data_table(
            data,
            impact_col,
            impacts,
            fibers,
            method=NORM_METHOD_ABS_MAX,
            fiber_cols=catalog_fibers,
        )
    )


def _chart_values_preview_panel_ui(
    table: pd.DataFrame,
    impact_col: str,
    *,
    normalized: bool,
    fiber_count: int,
    impact_count: int,
    panel_title: str = "Values table",
    error_message: str | None = None,
) -> ui.Tag:
    if error_message:
        body = ui.div(
            ui.tags.p(error_message, class_="lca-formula-lead mb-0 text-danger"),
            class_="lca-formula-body",
        )
        return _hover_panel_ui(
            trigger_label="Values table",
            icon_svg=_TABLE_ICON_SVG,
            title=panel_title,
            body=body,
        )

    if impact_count < 1:
        body = ui.div(
            ui.tags.p(
                "Select at least one environmental impact.",
                class_="lca-formula-lead mb-0",
            ),
            class_="lca-formula-body",
        )
    elif normalized:
        lead = (
            f"Per-impact normalized values (−1 to 1) for {impact_count} "
            f"impact(s) and {fiber_count} fiber(s). Each impact row is "
            "scaled independently."
        )
        body = ui.div(
            ui.tags.p(lead, class_="lca-formula-lead"),
            _compact_values_table_ui(table, impact_col, value_format="ratio"),
            class_="lca-formula-body",
        )
    else:
        fiber_note = (
            f"{fiber_count} fiber(s)"
            if fiber_count != 1
            else "single fiber selected"
        )
        body = ui.div(
            ui.tags.p(
                f"Raw values for {impact_count} impact(s) and {fiber_note}.",
                class_="lca-formula-lead",
            ),
            _compact_values_table_ui(table, impact_col, value_format="raw"),
            class_="lca-formula-body",
        )

    return _hover_panel_ui(
        trigger_label="Values table",
        icon_svg=_TABLE_ICON_SVG,
        title=panel_title,
        body=body,
    )


def _all_impacts_values_preview_tag(
    data: LcaData,
    impacts: list[str],
    fibers: list[str],
    *,
    norm_method: str,
) -> ui.Tag:
    normalized = _normalization_supported(len(fibers), method=norm_method)
    if normalized:
        table = _build_normalized_data_table(
            data.df,
            data.impact_col,
            impacts,
            fibers,
            method=NORM_METHOD_ABS_MAX,
        )
    else:
        table = _build_raw_radar_preview_table(
            data.df, data.impact_col, impacts, fibers
        )
    return _chart_values_preview_panel_ui(
        table,
        data.impact_col,
        normalized=normalized,
        fiber_count=len(fibers),
        impact_count=len(impacts),
    )


def _radar_norm_preview_tag(
    data: LcaData,
    impacts: list[str],
    fibers: list[str],
    *,
    norm_method: str,
) -> ui.Tag:
    return _all_impacts_values_preview_tag(
        data, impacts, fibers, norm_method=norm_method
    )


def _home_page_css() -> str:
    return f"""
.lca-home-page {{
    padding: 1.25rem 1.5rem 1.75rem;
}}
.lca-home-lead {{
    color: #495057;
    font-size: 0.95rem;
    margin-bottom: 1.25rem;
}}
.lca-home-layout {{
    display: flex;
    flex-direction: column;
    gap: 1.15rem;
}}
.lca-home-top-row {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    align-items: stretch;
}}
.lca-home-card.btn {{
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: flex-start;
    width: 100%;
    height: 100%;
    text-align: left;
    white-space: normal;
    border: 1px solid #cfe2ff;
    border-radius: 12px;
    background: {_GRADIENT_HOME_CARD};
    padding: 1.1rem 1.15rem;
    box-shadow: 0 1px 3px rgba(13, 110, 253, 0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease, background 0.15s ease;
}}
.lca-home-top-card.btn {{
    min-height: 11rem;
    padding: 1rem 0.95rem;
}}
.lca-home-card.btn:hover,
.lca-home-card.btn:focus {{
    transform: translateY(-2px);
    border-color: #9ec5fe;
    box-shadow: 0 8px 18px rgba(13, 110, 253, 0.14);
    background: {_GRADIENT_HOME_CARD_HOVER};
    color: inherit;
}}
.lca-home-card-title {{
    font-size: 1rem;
    font-weight: 700;
    color: #1a2b42;
    margin: 0 0 0.45rem 0;
}}
.lca-home-card-desc {{
    font-size: 0.84rem;
    line-height: 1.45;
    color: #495057;
    margin: 0;
}}
.lca-home-top-card .lca-home-card-title {{
    font-size: 0.95rem;
    line-height: 1.3;
}}
.lca-home-top-card .lca-home-card-desc {{
    font-size: 0.8rem;
    line-height: 1.45;
}}
.lca-home-group {{
    border: 1px solid #c5d9fc;
    border-radius: 12px;
    background: {_GRADIENT_HOME_GROUP};
    padding: 1.1rem 1.15rem 1.15rem;
    box-shadow: 0 1px 4px rgba(13, 110, 253, 0.07);
}}
.lca-home-group-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a2b42;
    margin: 0 0 0.35rem 0;
}}
.lca-home-group-desc {{
    font-size: 0.84rem;
    color: #495057;
    margin: 0 0 0.85rem 0;
}}
.lca-home-subcards {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    align-items: stretch;
}}
.lca-home-subcard.btn {{
    min-height: 7.25rem;
    background: {_GRADIENT_HOME_SUBCARD};
}}
.lca-home-subcard.btn:hover,
.lca-home-subcard.btn:focus {{
    background: {_GRADIENT_HOME_SUBCARD_HOVER};
}}
"""


def _home_nav_card(
    *,
    button_id: str,
    title: str,
    description: str,
    subcard: bool = False,
    top_row: bool = False,
) -> ui.Tag:
    card_class = "lca-home-card"
    if subcard:
        card_class += " lca-home-subcard"
    if top_row:
        card_class += " lca-home-top-card"
    return ui.input_action_button(
        button_id,
        ui.div(
            ui.tags.h6(title, class_="lca-home-card-title"),
            ui.tags.p(description, class_="lca-home-card-desc"),
        ),
        class_=card_class,
    )


def _home_nav_group(
    *,
    title: str,
    description: str,
    items: tuple[dict[str, str], ...] | list[dict[str, str]],
) -> ui.Tag:
    return ui.div(
        ui.tags.h5(title, class_="lca-home-group-title"),
        ui.tags.p(description, class_="lca-home-group-desc"),
        ui.div(
            *[
                _home_nav_card(
                    button_id=str(item["button_id"]),
                    title=str(item["title"]),
                    description=str(item["description"]),
                    subcard=True,
                )
                for item in items
            ],
            class_="lca-home-subcards",
        ),
        class_="lca-home-group",
    )


def _home_page_ui() -> ui.Tag:
    return ui.div(
        ui.tags.p(
            "This tool was designed to provide a comparison of select potential "
            "environmental impacts using Life Cycle Assessment frameworks and "
            "methodologies to compare selected fibers used in textile products. "
            "The information enables access to reliable data regarding these "
            "impacts. It is hoped that using this tool, designers, product "
            "developers and sourcing professionals can make informed decisions "
            "regarding raw materials selection.",
            class_="lca-home-lead",
        ),
        ui.tags.p(
            "In addition, the information can aid industry professionals in "
            "understanding their environmental footprint based on raw material "
            "production/cultivation processes. Unless one uses the specific "
            "production/cultivation and processing methods described, it is not "
            "intended to be used to communicate validated claims about those "
            "materials to consumers.",
            class_="lca-home-lead",
        ),
        ui.div(
            ui.div(
                _home_nav_card(
                    button_id="home_nav_comparison_table",
                    title="Comparison Table",
                    description="Side-by-side raw and normalized values for selected fibers and impacts.",
                    top_row=True,
                ),
                _home_nav_card(
                    button_id="home_nav_all_impacts",
                    title="All Potential Impact Factors",
                    description="Heatmap, radar, and bar charts across all selected impacts.",
                    top_row=True,
                ),
                _home_nav_card(
                    button_id="home_nav_method",
                    title="Methods",
                    description="Documentation on data sources and analysis methods.",
                    top_row=True,
                ),
                _home_nav_card(
                    button_id="home_nav_team",
                    title="Team & Contributions",
                    description="Project team members and acknowledgements.",
                    top_row=True,
                ),
                class_="lca-home-top-row",
            ),
            _home_nav_group(
                title="Impact factor sections",
                description="Bar charts focused on a related set of environmental impacts.",
                items=HOME_SECTION_CARDS,
            ),
            class_="lca-home-layout",
        ),
        class_="lca-home-page",
    )


def _home_nav_panel() -> ui.Tag:
    return ui.nav_panel(
        "Home",
        _with_partner_footer(
            ui.card(
                _home_page_ui(),
                full_screen=False,
            ),
            content_height=True,
        ),
        value=TAB_HOME,
    )


METHOD_PHASE_CARDS: tuple[dict[str, str], ...] = (
    {"title": "Goals and Scope", "subtitle": "Image and text TBA"},
    {"title": "LCI"},
    {"title": "LCIA"},
    {"title": "Interpretation"},
)


def _method_page_css() -> str:
    return """
.lca-method-page {
    padding: 1.25rem 1.5rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.15rem;
}
.lca-method-card {
    width: 100%;
    border: 1px solid #c5d9fc;
    border-radius: 12px;
    background: radial-gradient(
        ellipse 130% 120% at 50% 12%,
        #f7faff 0%,
        #edf3ff 40%,
        #e2ebff 100%
    );
    padding: 1.1rem 1.15rem 1.15rem;
    box-shadow: 0 1px 4px rgba(13, 110, 253, 0.07);
}
.lca-method-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a2b42;
    margin: 0 0 0.85rem 0;
    line-height: 1.3;
}
.lca-method-card-subtitle {
    font-size: 0.78rem;
    font-weight: 500;
    color: #6c757d;
    margin: -0.55rem 0 0.75rem 0;
    line-height: 1.35;
}
.lca-method-card-body {
    font-size: 0.84rem;
    color: #6c757d;
    line-height: 1.45;
    margin: 0;
}
"""


def _method_card_ui(card: dict[str, str]) -> ui.Tag:
    subtitle = card.get("subtitle")
    header_children: list[ui.Tag] = [
        ui.tags.h5(str(card["title"]), class_="lca-method-card-title"),
    ]
    if subtitle:
        header_children.append(
            ui.tags.p(str(subtitle), class_="lca-method-card-subtitle"),
        )
    return ui.tags.div(
        *header_children,
        ui.tags.p("Info will be added soon.", class_="lca-method-card-body"),
        class_="lca-method-card",
    )


def _method_page_ui() -> ui.Tag:
    return ui.div(
        *[_method_card_ui(card) for card in METHOD_PHASE_CARDS],
        class_="lca-method-page",
    )


def _method_nav_panel() -> ui.Tag:
    return ui.nav_panel(
        "Methods",
        _with_partner_footer(
            ui.card(
                _method_page_ui(),
                class_="lca-method-card-shell",
                full_screen=False,
            ),
            content_height=True,
        ),
        value=TAB_METHOD,
    )


def _team_page_css() -> str:
    return """
.lca-team-page {
    padding: 1.25rem 1.5rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.15rem;
}
.lca-team-function-card {
    border: 1px solid #c5d9fc;
    border-radius: 12px;
    background: radial-gradient(
        ellipse 130% 120% at 50% 12%,
        #f7faff 0%,
        #edf3ff 40%,
        #e2ebff 100%
    );
    padding: 1.1rem 1.15rem 1.15rem;
    box-shadow: 0 1px 4px rgba(13, 110, 253, 0.07);
}
.lca-team-function-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a2b42;
    margin: 0 0 0.85rem 0;
}
.lca-team-function-card--single {
    width: calc((100% - 0.85rem) / 2);
    max-width: 100%;
}
.lca-team-function-card--single .lca-team-org-subcards {
    grid-template-columns: 1fr;
}
.lca-team-org-subcards {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.85rem;
    align-items: start;
}
@media (max-width: 640px) {
    .lca-team-org-subcards {
        grid-template-columns: 1fr;
    }
    .lca-team-function-card--single {
        width: 100%;
    }
}
.lca-team-org-subcard {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 0.75rem;
    min-width: 0;
    border: 1px solid #cfe2ff;
    border-radius: 10px;
    background: #ffffff;
    padding: 0.85rem 0.8rem 0.9rem;
}
.lca-team-logo-link {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    min-height: 4.75rem;
    padding: 0.5rem 0.45rem;
    border: none;
    border-radius: 8px;
    background: transparent;
    box-shadow: none;
    cursor: pointer;
    text-decoration: none;
    color: inherit;
    position: relative;
    z-index: 2;
    pointer-events: auto;
    transition: opacity 0.15s ease, transform 0.15s ease;
}
.lca-team-logo-link:hover,
.lca-team-logo-link:focus {
    opacity: 0.82;
    transform: translateY(-1px);
    outline: none;
    text-decoration: none;
    color: inherit;
}
.lca-team-logo-link:active {
    transform: translateY(0);
    opacity: 0.72;
}
.lca-team-logo-img {
    display: block;
    max-width: 100%;
    max-height: 4rem;
    width: auto;
    height: auto;
    object-fit: contain;
}
.lca-team-member-list {
    margin: 0;
    padding-left: 1.15rem;
    font-size: 0.84rem;
    color: #343a40;
    line-height: 1.45;
}
.lca-team-member-list li {
    margin-bottom: 0.35rem;
}
.lca-team-member-list li:last-child {
    margin-bottom: 0;
}
.lca-team-ack-section {
    margin-top: 0.35rem;
    padding-top: 1rem;
    border-top: 1px solid #dee2e6;
}
.lca-team-ack {
    font-size: 0.82rem;
    color: #6c757d;
    line-height: 1.5;
    margin: 0;
    max-width: 52rem;
}
"""


def _logo_data_uri(filename: str) -> str:
    path = APP_ROOT / filename
    mime_by_ext = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime = mime_by_ext.get(path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _org_logo_link_ui(
    org: _TEAM_ORG_INFO,
    *,
    link_class: str,
    img_class: str,
) -> ui.Tag:
    return ui.tags.a(
        ui.tags.img(
            src=_logo_data_uri(str(org["logo"])),
            alt=str(org["name"]),
            class_=img_class,
        ),
        href=str(org["website"]),
        target="_blank",
        rel="noopener noreferrer",
        class_=link_class,
        title=str(org["name"]),
        **{"aria-label": f"Visit {org['name']} website (opens in new tab)"},
    )


def _team_logo_link_ui(org: _TEAM_ORG_INFO) -> ui.Tag:
    return _org_logo_link_ui(
        org,
        link_class="lca-team-logo-link",
        img_class="lca-team-logo-img",
    )


def _partner_footer_ui() -> ui.Tag:
    return ui.tags.footer(
        ui.div(
            *[
                _org_logo_link_ui(
                    TEAM_ORG_REGISTRY[org_id],
                    link_class="lca-partner-logo-link",
                    img_class="lca-partner-logo-img",
                )
                for org_id in PARTNER_ORG_ORDER
            ],
            class_="lca-partner-footer-logos",
        ),
        class_="lca-partner-footer",
        **{"aria-label": "Project partners"},
    )


def _with_partner_footer(main: ui.Tag, *, content_height: bool = False) -> ui.Tag:
    """Place partner logos after tab content in normal document flow."""
    page_class = "lca-tab-page"
    if content_height:
        page_class += " lca-tab-page--content"
    return ui.div(
        ui.div(main, class_="lca-tab-page-main"),
        _partner_footer_ui(),
        class_=page_class,
    )


def _team_member_line_ui(member: _TEAM_MEMBER) -> ui.Tag:
    name = str(member["name"])
    role = member.get("role")
    contributions = member.get("contributions")
    name_el = ui.tags.strong(name)
    if role and contributions:
        if isinstance(contributions, list):
            contrib_text = ", ".join(str(item) for item in contributions)
        else:
            contrib_text = str(contributions)
        return ui.tags.li(name_el, f": {role}, {contrib_text}")
    if role:
        return ui.tags.li(name_el, f": {role}")
    return ui.tags.li(name_el)


def _team_org_subcard_ui(org_group: _TEAM_ORG_GROUP) -> ui.Tag:
    org_id = str(org_group["org_id"])
    org = TEAM_ORG_REGISTRY[org_id]
    members = org_group["members"]
    member_list = members if isinstance(members, list) else []
    return ui.tags.div(
        _team_logo_link_ui(org),
        ui.tags.ul(
            *[_team_member_line_ui(member) for member in member_list],
            class_="lca-team-member-list",
        ),
        class_="lca-team-org-subcard",
    )


def _team_function_card_ui(area: _TEAM_FUNCTION_AREA) -> ui.Tag:
    orgs = area["orgs"]
    org_list = orgs if isinstance(orgs, list) else []
    card_class = "lca-team-function-card"
    if len(org_list) == 1:
        card_class += " lca-team-function-card--single"
    return ui.tags.div(
        ui.tags.h5(str(area["title"]), class_="lca-team-function-title"),
        ui.tags.div(
            *[_team_org_subcard_ui(org_group) for org_group in org_list],
            class_="lca-team-org-subcards",
        ),
        class_=card_class,
    )


def _team_page_ui() -> ui.Tag:
    return ui.div(
        *[_team_function_card_ui(area) for area in TEAM_FUNCTION_AREAS],
        ui.div(
            ui.tags.p(
                "This project was supported by a collaboration between the "
                "Walmart Foundation, the Wilson College of Textiles at NC "
                "State University, the National Laboratory of the Rockies, "
                "and The Textile Innovation Engine.",
                class_="lca-team-ack",
            ),
            class_="lca-team-ack-section",
        ),
        class_="lca-team-page",
    )


def _team_nav_panel() -> ui.Tag:
    return ui.nav_panel(
        "Team & Contributions",
        _with_partner_footer(
            ui.card(
                ui.card_header("We would like to thank the Walmart Foundation for its financial support of this project."),
                _team_page_ui(),
                class_="lca-team-card",
                full_screen=False,
            ),
            content_height=True,
        ),
        value=TAB_TEAM,
    )


def _references_nav_panel() -> ui.Tag:
    return ui.nav_panel(
        "References",
        _with_partner_footer(
            ui.card(
                ui.div(
                    ui.tags.p(
                        "Info will be added soon.",
                        class_="text-muted px-3 py-3 mb-0",
                    ),
                ),
                full_screen=False,
            ),
            content_height=True,
        ),
        value=TAB_REFERENCES,
    )


def _section_nav_panel(section_key: str) -> ui.Tag:
    spec = IMPACT_SECTIONS[section_key]
    title = str(spec["tab"])
    return ui.nav_panel(
        title,
        _with_partner_footer(
            ui.card(
                ui.output_ui(f"section_note_{section_key}"),
                ui.output_ui(f"section_toolbar_{section_key}"),
                _chart_plot_output(f"section_plot_{section_key}"),
                full_screen=True,
            ),
        ),
        value=section_key,
    )


def _picker_summary_label(
    selected: list[str],
    choices: list[str],
    *,
    placeholder: str,
) -> str:
    if not choices:
        return "No options available"
    if not selected:
        return placeholder
    if set(selected) >= set(choices):
        return f"All selected ({len(choices)})"
    if len(selected) == 1:
        return selected[0]
    return f"{len(selected)} of {len(choices)} selected"


def _render_picker_summary(
    selected: list[str],
    choices: list[str],
    *,
    placeholder: str,
) -> ui.Tag:
    text = _picker_summary_label(selected, choices, placeholder=placeholder)
    cls = "lca-picker-summary-text"
    if not selected:
        cls += " lca-picker-summary-placeholder"
    return ui.tags.span(text, class_=cls)


def _checkbox_dropdown_panel(
    input_id: str,
    title: str,
    choices: list[str],
    *,
    all_switch_id: str,
    summary_output_id: str,
    selected: list[str] | None = None,
    max_height: str | None = None,
    placeholder: str = "Click to select…",
) -> ui.Tag:
    """Collapsed select-style trigger; opens a panel with checkboxes."""
    selected_list = selected if selected is not None else choices
    all_on = bool(choices) and set(selected_list) >= set(choices)

    group = ui.input_checkbox_group(
        input_id,
        "",
        choices,
        selected=selected_list,
    )
    menu_class = "lca-checkbox-dropdown-menu"
    if max_height:
        menu_class += " lca-checkbox-scroll"
        menu_style = f"--lca-scroll-h: {max_height};"
    else:
        menu_style = None

    menu = ui.tags.div(
        group,
        class_=menu_class,
        style=menu_style,
    )

    header = ui.tags.div(
        ui.tags.span(title, class_="lca-checkbox-panel-title"),
        ui.tags.div(
            ui.input_switch(all_switch_id, "All", value=all_on),
            class_="lca-all-switch-wrap",
        ),
        class_="lca-checkbox-card-header",
    )
    body = ui.tags.details(
        ui.tags.summary(
            ui.output_ui(summary_output_id),
            class_="lca-checkbox-dropdown-trigger",
        ),
        menu,
        class_="lca-checkbox-details",
        **{"data-lca-picker": input_id},
    )
    return ui.card(header, body, class_="lca-checkbox-card")


def _select_card_panel(
    input_id: str,
    title: str,
    choices: list[str],
    *,
    selected: str | None = None,
    multiple: bool = False,
) -> ui.Tag:
    """Single-select input wrapped in the same card frame as checkbox dropdown panels."""
    header = ui.tags.div(
        ui.tags.span(title, class_="lca-checkbox-panel-title"),
        class_="lca-checkbox-card-header",
    )
    body = ui.tags.div(
        ui.input_selectize(
            input_id,
            "",
            choices=choices,
            selected=selected,
            multiple=multiple,
        ),
        class_="lca-select-card-body",
    )
    return ui.card(header, body, class_="lca-checkbox-card")


_CHECKBOX_PANEL_CSS = ("""
.lca-checkbox-card {
    margin-bottom: 0.5rem !important;
}
.lca-checkbox-card .card-body {
    padding: 0;
    overflow: visible;
}
.lca-checkbox-details {
    margin: 0;
}
.lca-checkbox-details > summary {
    display: flex;
    align-items: center;
    list-style: none;
    cursor: pointer;
    margin: 0.45rem 0.85rem 0.5rem;
    padding: 0.45rem 2rem 0.45rem 0.65rem;
    min-height: 2.25rem;
    border: 1px solid #ced4da;
    border-radius: 0.375rem;
    background: #fff;
    font-size: 0.875rem;
    color: #212529;
    position: relative;
}
.lca-checkbox-details > summary::-webkit-details-marker,
.lca-checkbox-details > summary::marker {
    display: none;
    content: "";
}
.lca-checkbox-details > summary::after {
    content: "";
    position: absolute;
    right: 0.75rem;
    top: 50%;
    width: 0.45rem;
    height: 0.45rem;
    margin-top: -0.3rem;
    border-right: 2px solid #6c757d;
    border-bottom: 2px solid #6c757d;
    transform: rotate(45deg);
    pointer-events: none;
}
.lca-checkbox-details[open] > summary {
    border-color: #86b7fe;
    box-shadow: 0 0 0 0.2rem rgba(13, 110, 253, 0.15);
    margin-bottom: 0.35rem;
}
.lca-checkbox-dropdown-menu {
    margin: 0 1rem 0.85rem;
    padding: 0.55rem 0.65rem;
    border: 1px solid #dee2e6;
    border-radius: 0.375rem;
    background: #fff;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.12);
}
.lca-checkbox-dropdown-menu .shiny-input-checkboxgroup .form-check {
    margin-bottom: 0.35rem;
}
.lca-checkbox-dropdown-menu .shiny-input-checkboxgroup .form-check:last-child {
    margin-bottom: 0;
}
.lca-picker-summary-text {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding-right: 0.25rem;
}
.lca-picker-summary-placeholder {
    color: #6c757d;
}
.lca-checkbox-scroll {
    max-height: var(--lca-scroll-h, 320px);
    overflow-x: hidden;
    overflow-y: scroll;
    padding: 0.85rem 1rem 0.75rem;
    scrollbar-gutter: stable;
    scrollbar-width: thin;
}
.lca-checkbox-scroll::-webkit-scrollbar {
    width: 10px;
}
.lca-checkbox-scroll::-webkit-scrollbar-thumb {
    background: #b8b8b8;
    border-radius: 6px;
    border: 2px solid #f8f9fa;
}
.lca-checkbox-scroll::-webkit-scrollbar-track {
    background: #f0f0f0;
    border-radius: 6px;
}
.lca-checkbox-scroll .shiny-input-checkboxgroup {
    margin-bottom: 0;
}
.lca-checkbox-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.85rem;
    background: """
    + _GRADIENT_SIDEBAR_HEADER
    + """;
    border-bottom: 1px solid #dbeafe;
}
.lca-checkbox-panel-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #0a58ca;
}
.lca-all-switch-wrap {
    display: inline-flex;
    align-items: center;
    flex-shrink: 0;
}
.lca-all-switch-wrap .form-check {
    margin: 0;
    padding: 0;
    min-height: 0;
    display: inline-flex;
    align-items: center;
}
.lca-all-switch-wrap .form-switch {
    margin: 0;
    min-height: 0;
    padding-left: 2.25em;
    display: inline-flex;
    align-items: center;
}
.lca-all-switch-wrap .form-check-input {
    width: 1.85rem;
    height: 1rem;
    margin-left: -2.25em;
    margin-top: 0;
    cursor: pointer;
    border: 1px solid #adb5bd;
    background-color: #e9ecef;
    float: none;
}
.lca-all-switch-wrap .form-check-input:checked {
    background-color: #0d6efd;
    border-color: #0d6efd;
}
.lca-all-switch-wrap .form-check-input:focus {
    box-shadow: 0 0 0 0.12rem rgba(13, 110, 253, 0.25);
}
.lca-all-switch-wrap .form-check-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #495057;
    line-height: 1;
    padding-left: 0.25rem;
    margin: 0;
    cursor: pointer;
}
.lca-select-card-body {
    padding: 0.65rem 1rem 0.75rem;
}
.lca-select-card-body .control-label {
    display: none;
}
.lca-select-card-body .shiny-input-container {
    margin: 0;
}
.lca-select-card-body .selectize-control.single .selectize-input {
    min-height: 2.25rem;
    padding: 0.45rem 2rem 0.45rem 0.65rem;
    border: 1px solid #ced4da;
    border-radius: 0.375rem;
    background: #fff;
    font-size: 0.875rem;
    color: #212529;
    box-shadow: none;
}
.lca-select-card-body .selectize-control.single .selectize-input.focus {
    border-color: #86b7fe;
    box-shadow: 0 0 0 0.2rem rgba(13, 110, 253, 0.15);
}
.lca-select-card-body .selectize-control.single .selectize-input:after {
    right: 0.75rem;
    width: 0.45rem;
    height: 0.45rem;
    margin-top: -0.3rem;
    border: none;
    border-right: 2px solid #6c757d;
    border-bottom: 2px solid #6c757d;
    transform: rotate(45deg);
    background: none;
}
.lca-formula-trigger {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    border: 1px solid #0d6efd;
    background: linear-gradient(180deg, #f8fbff 0%, #e8f2ff 100%);
    color: #0a58ca;
    font-size: 0.875rem;
    font-weight: 600;
    box-shadow: 0 1px 2px rgba(13, 110, 253, 0.15);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.lca-formula-trigger:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(13, 110, 253, 0.22);
    background: linear-gradient(180deg, #ffffff 0%, #dbeafe 100%);
}
.lca-formula-popover.popover {
    border: none;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18);
    max-width: 520px;
}
.lca-formula-popover .popover-header {
    background: linear-gradient(135deg, #0d6efd 0%, #0a58ca 100%);
    color: #fff;
    font-weight: 600;
    border-bottom: none;
    padding: 0.75rem 1rem;
}
.lca-formula-popover .popover-body {
    padding: 0;
}
.lca-formula-body {
    padding: 1rem 1.1rem 1.05rem;
    background: #fafbff;
}
.lca-formula-lead {
    color: #212529;
    font-size: 0.875rem;
    margin-bottom: 0.75rem;
}
.lca-formula-equation {
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 0.95rem;
    line-height: 1.55;
    padding: 0.65rem 0.75rem;
    margin-bottom: 0.85rem;
    border-radius: 8px;
    background: #fff;
    border: 1px solid #dbe4f0;
    color: #1a2b42;
    text-align: left;
}
.lca-formula-equation-fraction {
    display: block;
    margin: 0.35rem 0 0.35rem 0.5rem;
    padding-left: 0.5rem;
    border-left: 2px solid #dbe4f0;
}
.lca-formula-var {
    color: #0d6efd;
    font-weight: 600;
}
.lca-formula-legend {
    list-style: none;
    padding: 0;
    margin: 0;
    font-size: 0.875rem;
    color: #212529;
}
.lca-formula-legend li {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    margin-bottom: 0.45rem;
}
.lca-formula-legend li:last-child {
    margin-bottom: 0;
}
.lca-formula-badge {
    flex-shrink: 0;
    min-width: 2.5rem;
    text-align: center;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 0.15rem 0.4rem;
    border-radius: 6px;
    background: #e7f1ff;
    color: #0a58ca;
}
.lca-section-impacts {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem 0.5rem;
}
.lca-section-impact-chip {
    display: inline-block;
    font-size: 0.8125rem;
    font-weight: 600;
    line-height: 1.35;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    background: #e7f1ff;
    color: #0a58ca;
    border: 1px solid #b6d4fe;
}
.lca-section-impact-chip-muted {
    background: #f0f4f8;
    color: #6c757d;
    border-color: #dee2e6;
}
.lca-norm-popover.popover {
    max-width: min(92vw, 540px);
}
.lca-norm-table-wrap {
    max-width: 100%;
    max-height: min(52vh, 380px);
    overflow: auto;
    border-radius: 8px;
    border: 1px solid #dbe4f0;
    background: #fff;
    scrollbar-gutter: stable;
    scrollbar-width: thin;
}
.lca-norm-table-wrap::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
.lca-norm-table-wrap::-webkit-scrollbar-thumb {
    background: #b6c2d1;
    border-radius: 4px;
}
.lca-norm-table {
    font-size: 0.78rem;
    margin-bottom: 0;
    width: max-content;
    min-width: 100%;
}
.lca-norm-table thead th {
    position: sticky;
    top: 0;
    background: #eef4ff;
    color: #1a2b42;
    font-weight: 600;
    white-space: nowrap;
    z-index: 1;
}
.lca-norm-table-impact {
    min-width: 9rem;
    max-width: 12rem;
}
.lca-norm-table-fiber {
    text-align: right;
    min-width: 3.5rem;
}
.lca-norm-table td.text-end {
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    color: #0a58ca;
}
.lca-missing-value {
    color: #c0392b !important;
    font-weight: 700;
}
.lca-heatmap-missing-sample {
    display: inline-block;
    width: 1.15em;
    height: 1.15em;
    background: #fff;
    border: 1px solid #e0e0e0;
    vertical-align: -0.15em;
    margin-right: 0.2em;
    position: relative;
    overflow: hidden;
}
.lca-heatmap-missing-sample::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(
        to top right,
        transparent 0,
        transparent calc(50% - 0.5px),
        #aaa calc(50% - 0.5px),
        #aaa calc(50% + 0.5px),
        transparent calc(50% + 0.5px),
        transparent 100%
    );
}
.lca-hover-panel {
    position: relative;
    display: inline-block;
}
.lca-hover-panel-dropdown {
    display: none;
    position: absolute;
    left: 0;
    top: 100%;
    padding-top: 4px;
    z-index: 1060;
    min-width: min(92vw, 340px);
    max-width: min(92vw, 720px);
    width: min(92vw, 720px);
}
.lca-hover-panel:hover .lca-hover-panel-dropdown {
    display: block;
}
.lca-hover-panel-card {
    border-radius: 10px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18);
    overflow: hidden;
    background: #fafbff;
    max-height: min(75vh, 560px);
    display: flex;
    flex-direction: column;
}
.lca-hover-panel-title {
    background: linear-gradient(135deg, #0d6efd 0%, #0a58ca 100%);
    color: #fff;
    font-weight: 600;
    padding: 0.65rem 1rem;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.lca-hover-panel-body {
    padding: 0;
    overflow: auto;
    max-height: min(68vh, 500px);
    overscroll-behavior: contain;
    scrollbar-gutter: stable;
    scrollbar-width: thin;
}
.lca-hover-panel-body::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
.lca-hover-panel-body::-webkit-scrollbar-thumb {
    background: #b6c2d1;
    border-radius: 4px;
}
.lca-norm-method-panel {
    padding: 1rem 1.1rem 1.05rem;
    max-width: 100%;
    box-sizing: border-box;
}
.lca-norm-method-panel .lca-norm-formula-block {
    max-width: 100%;
    overflow-x: auto;
    overflow-y: visible;
}
.lca-norm-method-panel .shiny-input-radiogroup,
.lca-norm-method-panel .shiny-input-checkboxgroup {
    margin-bottom: 0.85rem;
}
.lca-norm-method-panel .shiny-input-radiogroup > label.control-label {
    font-size: 0.875rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: block;
}
.lca-norm-method-panel .shiny-options-group,
.lca-norm-radio-options .shiny-options-group {
    margin-top: 0.35rem;
}
.lca-norm-method-panel .shiny-options-group .radio {
    margin-bottom: 0.3rem;
}
.lca-norm-method-panel .shiny-options-group .radio label,
.lca-norm-method-panel .shiny-options-group .radio label span,
.lca-norm-radio-options .shiny-options-group .radio label,
.lca-norm-radio-options .shiny-options-group .radio label span {
    font-size: 0.68rem !important;
    line-height: 1.35 !important;
    font-weight: 400 !important;
}
.lca-norm-formula-block {
    margin-top: 0.25rem;
    color: #212529;
}
.lca-norm-formula-block .lca-formula-lead,
.lca-norm-formula-block .lca-formula-legend {
    color: #212529;
}
.lca-formula-guide-table-wrap {
    margin-top: 0.5rem;
}
.lca-formula-guide-table {
    margin-bottom: 0;
    font-size: 0.84rem;
}
.lca-formula-guide-table thead th {
    background: #eef4fc;
    color: #1a2b42;
    font-weight: 600;
}
.lca-formula-guide-table td.text-end {
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
""")


def _hover_panel_ui(
    *,
    trigger_label: str,
    icon_svg: str,
    title: str,
    body: ui.Tag,
) -> ui.Tag:
    """Dropdown stays open while the pointer is on the button or panel (scrollable)."""
    return ui.tags.div(
        ui.tags.button(
            ui.HTML(icon_svg),
            ui.tags.span(trigger_label),
            type="button",
            class_="lca-formula-trigger",
        ),
        ui.tags.div(
            ui.tags.div(
                ui.tags.div(title, class_="lca-hover-panel-title"),
                ui.tags.div(body, class_="lca-hover-panel-body"),
                class_="lca-hover-panel-card",
            ),
            class_="lca-hover-panel-dropdown",
        ),
        class_="lca-hover-panel",
    )


def _hover_popover(
    *,
    trigger_label: str,
    icon_svg: str,
    title: str,
    body: ui.Tag,
    popover_id: str,
    extra_class: str = "",
) -> ui.Tag:
    popover_class = "lca-formula-popover"
    if extra_class:
        popover_class = f"{popover_class} {extra_class}"
    trigger = ui.tags.button(
        ui.HTML(icon_svg),
        ui.tags.span(trigger_label),
        type="button",
        class_="lca-formula-trigger",
        title=title,
    )
    return ui.popover(
        trigger,
        body,
        title=title,
        placement="right",
        id=popover_id,
        options={
            "customClass": popover_class,
            "trigger": "hover focus",
            "html": True,
        },
    )


def _compact_values_table_ui(
    table: pd.DataFrame,
    impact_col: str,
    *,
    value_format: str = "raw",
) -> ui.Tag:
    if table.empty:
        return ui.tags.p("No rows for the current selection.", class_="mb-0 small text-muted")

    fiber_cols = [c for c in table.columns if c not in {impact_col, "Units"}]
    header = ui.tags.tr(
        ui.tags.th("Impact", class_="lca-norm-table-impact"),
        *[
            ui.tags.th(_short_theta_label(f, 18), class_="lca-norm-table-fiber")
            for f in fiber_cols
        ],
    )
    rows = []
    for _, row in table.iterrows():
        cells = [
            ui.tags.td(_short_theta_label(str(row[impact_col]), 34), class_="lca-norm-table-impact")
        ]
        for fiber in fiber_cols:
            value = row[fiber]
            if _json_safe_number(value) is None:
                cells.append(
                    ui.tags.td(
                        MISSING_VALUE_LABEL,
                        class_="text-end lca-missing-value",
                    )
                )
            else:
                if value_format == "percent":
                    text = _format_display_number(value, suffix="%")
                else:
                    text = _format_display_number(value)
                cells.append(ui.tags.td(text, class_="text-end"))
        rows.append(ui.tags.tr(*cells))

    return ui.tags.div(
        ui.tags.table(
            ui.tags.thead(header),
            ui.tags.tbody(*rows),
            class_="table table-sm table-hover lca-norm-table",
        ),
        class_="lca-norm-table-wrap",
    )


def _comparison_table_section(title: str, output_id: str) -> ui.Tag:
    return ui.div(
        ui.div(
            ui.tags.h6(title, class_="mb-0"),
            _table_download_buttons(output_id),
            class_="d-flex justify-content-between align-items-center px-3 pt-3 mb-1 gap-2",
        ),
        ui.output_data_frame(output_id),
    )


def _norm_formula_equation_ui(method: str) -> ui.Tag:
    if method == NORM_METHOD_RAW:
        return ui.div(
            ui.tags.p(
                "No normalization is applied. Each value is shown as stored in "
                "the dataset for that fiber and impact category (see the Units "
                "column where available).",
                class_="lca-formula-lead",
            ),
        )

    return ui.div(
        ui.tags.p(
            "Each impact category is scaled separately. Every fiber value "
            "in that row is divided by the largest absolute value among "
            "the compared fibers (range −1 to 1).",
            class_="lca-formula-lead",
        ),
        ui.tags.div(
            ui.tags.strong("Normalized value"),
            " = ",
            ui.tags.span("Original value", class_="lca-formula-var"),
            " ÷ ",
            ui.tags.span(
                "Maximum absolute value within that impact category",
                class_="lca-formula-var",
            ),
            class_="lca-formula-equation",
        ),
        ui.tags.ul(
            ui.tags.li(
                ui.tags.span("1", class_="lca-formula-badge"),
                ui.tags.span("Highest absolute value in that impact category"),
            ),
            ui.tags.li(
                ui.tags.span("−1", class_="lca-formula-badge"),
                ui.tags.span(
                    "Largest negative value relative to the max absolute "
                    "in that impact category"
                ),
            ),
            ui.tags.li(
                ui.tags.span("0", class_="lca-formula-badge"),
                ui.tags.span("Zero impact for that category"),
            ),
            class_="lca-formula-legend",
        ),
    )


def _normalization_hover_panel_ui(
    *,
    method_input_id: str,
    formula_output_id: str,
    selected_method: str = NORM_METHOD_ABS_MAX,
) -> ui.Tag:
    """Hover panel: radio buttons to pick raw vs per-impact normalized data."""
    body = ui.div(
        ui.tags.p(
            "Choose raw values or per-impact normalized data (−1 to 1). "
            "Each impact category is scaled independently.",
            class_="lca-formula-lead mb-2",
        ),
        ui.div(
            ui.input_radio_buttons(
                method_input_id,
                "Normalization method",
                choices=NORM_METHOD_CHOICES,
                selected=selected_method,
            ),
            class_="lca-norm-radio-options",
        ),
        ui.output_ui(formula_output_id),
        class_="lca-formula-body lca-norm-method-panel",
    )
    return _hover_panel_ui(
        trigger_label="Normalization",
        icon_svg=_FORMULA_ICON_SVG,
        title="Normalization method & formula",
        body=body,
    )


def _chart_guide_body(
    *items: ui.Tag,
    lead: ui.Tag | None = None,
    lead_class: str = "lca-formula-lead mb-2",
    list_class: str = "lca-formula-legend mb-0",
) -> ui.Tag:
    if lead is None:
        lead = ui.tags.p("How to read this chart:", class_=lead_class)
    return ui.div(
        lead,
        ui.tags.ul(*items, class_=list_class),
        class_="lca-formula-body",
    )


def _chart_guide_panel(trigger_label: str, title: str, body: ui.Tag) -> ui.Tag:
    return _hover_panel_ui(
        trigger_label=trigger_label,
        icon_svg=_GUIDE_ICON_SVG,
        title=title,
        body=body,
    )


def _radar_guide_panel_ui() -> ui.Tag:
    return _chart_guide_panel(
        "Radar guide",
        "How the radar chart works",
        _chart_guide_body(
            ui.tags.li(
                ui.tags.strong("Vertices"),
                ": each selected environmental impact.",
            ),
            ui.tags.li(
                ui.tags.strong("Filled areas"),
                ": one fiber per color (see legend).",
            ),
            ui.tags.li("Hover a point for raw and normalized values."),
            lead_class="lca-formula-lead",
            list_class="lca-formula-legend",
        ),
    )


def _all_impacts_vertical_bar_guide_panel_ui() -> ui.Tag:
    return _chart_guide_panel(
        "Bar Chart guide",
        "Vertical bar chart",
        _chart_guide_body(
            ui.tags.li(
                "With one impact selected, bars compare fibers side by side. "
                "With multiple impacts, fibers are grouped into bundles per "
                "impact category on a single chart (see the legend for fiber colors).",
            ),
            ui.tags.li(
                ui.tags.strong("Gray bars"),
                " mean that fiber has no value (N/A) for that impact.",
            ),
        ),
    )


def _all_impacts_horizontal_bar_guide_panel_ui() -> ui.Tag:
    return _chart_guide_panel(
        "Bar Chart guide",
        "Horizontal bar chart",
        _chart_guide_body(
            ui.tags.li(
                "Bars extend horizontally: impacts (or fibers when one impact "
                "is selected) are listed on the vertical axis and values on "
                "the horizontal axis.",
            ),
            ui.tags.li(
                "With multiple impacts, fibers are grouped per impact category "
                "(see the legend for fiber colors).",
            ),
            ui.tags.li(
                ui.tags.strong("Gray bars"),
                " mean that fiber has no value (N/A) for that impact.",
            ),
        ),
    )


def _bar_chart_guide_panel_ui() -> ui.Tag:
    return _chart_guide_panel(
        "Bar Chart guide",
        "Bar chart",
        _chart_guide_body(
            ui.tags.li(
                "One panel per selected environmental impact; bars compare "
                "the chosen fibers side by side.",
            ),
            ui.tags.li(
                ui.tags.strong("Red fiber labels"),
                " below a bar mean that fiber has no value (N/A) for that impact.",
            ),
        ),
    )


def _heatmap_guide_panel_ui() -> ui.Tag:
    return _chart_guide_panel(
        "Heatmap guide",
        "Heatmap",
        _chart_guide_body(
            ui.tags.li("Each row scaled separately (impacts use different units)."),
            ui.tags.li(
                "Light to dark blue = lowest to highest value within that impact row.",
            ),
            ui.tags.li(
                ui.tags.span(
                    class_="lca-heatmap-missing-sample",
                    **{"aria-hidden": "true"},
                ),
                " = missing or empty value in the dataset.",
            ),
            ui.tags.li("Hover a cell for the full name and exact value."),
            lead=ui.tags.p(
                ui.tags.strong("Impact × Fiber heatmap (raw values)"),
                class_="lca-formula-lead mb-2",
            ),
        ),
    )


# ── App UI ───────────────────────────────────────────────────────────────────

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.style(
            _CHECKBOX_PANEL_CSS
            + _section_tab_nav_css()
            + _apply_button_css()
            + _sidebar_note_css()
            + _app_title_css()
            + _partner_footer_css()
            + _plot_card_css()
            + _home_page_css()
            + _method_page_css()
            + _team_page_css()
        ),
        _checkbox_dropdown_panel(
            "fiber",
            "Fiber Type",
            INITIAL_DATA.fiber_cols,
            all_switch_id="fiber_all",
            summary_output_id="fiber_picker_summary",
            selected=[],
            placeholder="Click to select fiber types…",
        ),
        _checkbox_dropdown_panel(
            "impact",
            "Environmental Impact",
            INITIAL_DATA.impact_choices,
            all_switch_id="impact_all",
            summary_output_id="impact_picker_summary",
            selected=[],
            max_height="280px",
            placeholder="Click to select environmental impacts…",
        ),
        ui.input_action_button(
            "apply_settings",
            "Apply",
            class_="lca-apply-btn",
        ),
        _sidebar_methods_note_ui(),
        ui.tags.script("""
(function () {
  function flushPicker(details) {
    var picker = details.getAttribute("data-lca-picker");
    if (picker && window.Shiny) {
      Shiny.setInputValue(
        "picker_flush",
        { picker: picker, t: Date.now() },
        { priority: "event" }
      );
    }
  }

  document.querySelectorAll(".lca-checkbox-details[data-lca-picker]").forEach(function (el) {
    el.addEventListener("toggle", function () {
      if (!el.open) {
        flushPicker(el);
      }
    });
  });

  document.addEventListener("click", function (e) {
    var applyBtn = e.target.closest && e.target.closest("button#apply_settings");
    if (applyBtn) {
      document.querySelectorAll(".lca-checkbox-details[open]").forEach(function (el) {
        el.removeAttribute("open");
        flushPicker(el);
      });
      return;
    }
    document.querySelectorAll(".lca-checkbox-details[open]").forEach(function (el) {
      if (!el.contains(e.target)) {
        el.removeAttribute("open");
        flushPicker(el);
      }
    });
  });
})();
"""),
        width=360,
    ),
    ui.navset_tab(
        _home_nav_panel(),
        ui.nav_panel(
            "Comparision Table",
            _with_partner_footer(
                ui.card(
                    _comparison_table_section("Raw data", "data_tab_raw"),
                    _comparison_table_section(
                        "Normalized data", "data_tab_normalized"
                    ),
                    full_screen=True,
                ),
            ),
            value=TAB_COMPARISON,
        ),
        ui.nav_panel(
            "All Potential Impact Factors",
            _with_partner_footer(
                ui.card(
                    ui.input_select(
                        "chart_kind_all",
                        "Chart type",
                        choices=_all_impacts_chart_kind_choices(),
                        selected="radar",
                    ),
                    ui.output_ui("all_impacts_toolbar"),
                    _chart_plot_output("all_impacts_plot"),
                    full_screen=True,
                ),
            ),
            value=TAB_ALL_IMPACTS,
        ),
        *[
            _section_nav_panel(section_key)
            for section_key in IMPACT_SECTIONS
        ],
        _method_nav_panel(),
        _team_nav_panel(),
        _references_nav_panel(),
        id="main_tabs",
    ),
    title=ui.input_action_button(
        "nav_home_title",
        APP_TITLE,
        class_="lca-app-title-btn",
    ),
    fillable=True,
)

# ── Server ───────────────────────────────────────────────────────────────────

def server(input, output, session):
    data_store = reactive.Value(load_processed_data(DATA_FILE))

    @reactive.calc
    def current_data() -> LcaData:
        return data_store.get()

    def _live_fibers() -> list[str]:
        return list(input.fiber() or [])

    def _live_impacts() -> list[str]:
        return list(input.impact() or [])

    _applied_state = reactive.Value(_initial_applied_state())
    _norm_method = reactive.Value(NORM_METHOD_ABS_MAX)
    _live_fiber_cache = reactive.Value([])
    _live_impact_cache = reactive.Value([])
    _locking_section_impacts = reactive.Value(False)

    @reactive.effect
    @reactive.event(input.fiber)
    def _cache_live_fibers() -> None:
        _live_fiber_cache.set(list(input.fiber() or []))

    @reactive.effect
    @reactive.event(input.impact)
    def _cache_live_impacts() -> None:
        _live_impact_cache.set(list(input.impact() or []))

    def _capture_applied_state() -> AppliedState:
        data = current_data()
        section = _section_active.get()
        if section is not None:
            allowed = set(_section_impact_names(section, data.impact_choices))
            impacts = [i for i in _live_impact_cache.get() if i in allowed]
        else:
            impacts = list(_live_impact_cache.get())
        return AppliedState(
            fibers=list(_live_fiber_cache.get()),
            impacts=impacts,
        )

    def _commit_applied_state(state: AppliedState) -> None:
        _applied_state.set(
            AppliedState(
                fibers=list(state.fibers),
                impacts=list(state.impacts),
            )
        )

    @reactive.effect
    @reactive.event(input.apply_settings)
    def _on_apply_settings() -> None:
        _live_fiber_cache.set(list(input.fiber() or []))
        _live_impact_cache.set(list(input.impact() or []))
        state = _capture_applied_state()
        _commit_applied_state(state)

    def _comparison_table_for_method(norm_method: str) -> pd.DataFrame:
        state = _applied_state.get()
        data = current_data()
        if data.df.empty:
            return pd.DataFrame(
                {"Message": [f"Data file not found or empty: {DATA_FILE}"]}
            )
        if not state.fibers or not state.impacts:
            return pd.DataFrame(
                {"Message": ["Select at least one fiber and one impact."]}
            )
        return _build_chart_data_table(
            list(state.impacts),
            list(state.fibers),
            data.df,
            data.impact_col,
            norm_method=norm_method,
            fiber_cols=data.fiber_cols,
        )

    @reactive.calc
    def comparison_table_raw_df() -> pd.DataFrame:
        return _comparison_table_for_method(NORM_METHOD_RAW)

    @reactive.calc
    def comparison_table_normalized_df() -> pd.DataFrame:
        return _comparison_table_for_method(NORM_METHOD_PER_IMPACT)

    _applying_fiber_all = reactive.Value(False)
    _applying_impact_all = reactive.Value(False)
    _syncing_fiber_all_switch = reactive.Value(False)
    _syncing_impact_all_switch = reactive.Value(False)
    _syncing_norm_method = reactive.Value(False)
    _impacts_before_section = reactive.Value[list[str] | None](None)
    _section_active = reactive.Value[str | None](None)

    def _show_section_impact_lock_notice() -> None:
        ui.notification_show(
            SECTION_IMPACT_LOCK_MSG,
            type="warning",
            duration=4.5,
        )

    def _section_locked_impacts(section_key: str) -> list[str]:
        data = current_data()
        return _section_impact_names(section_key, data.impact_choices)

    def _section_chart_impacts(section_key: str) -> list[str]:
        state = _applied_state.get()
        catalog = _section_locked_impacts(section_key)
        selected = set(state.impacts)
        return [impact for impact in catalog if impact in selected]

    def _update_section_impact_sidebar(
        section_key: str,
        *,
        selected: list[str] | None = None,
    ) -> None:
        section_impacts = _section_locked_impacts(section_key)
        allowed = set(section_impacts)
        if selected is None:
            picked = [i for i in (input.impact() or []) if i in allowed]
        else:
            picked = [i for i in selected if i in allowed]
        _locking_section_impacts.set(True)
        try:
            ui.update_checkbox_group(
                "impact",
                choices=section_impacts,
                selected=picked,
            )
            ui.update_switch(
                "impact_all",
                value=bool(section_impacts) and set(picked) == set(section_impacts),
            )
        finally:
            _locking_section_impacts.set(False)

    def _enter_section_impact_sidebar(section_key: str) -> None:
        _update_section_impact_sidebar(
            section_key,
            selected=_section_locked_impacts(section_key),
        )

    @reactive.calc
    def active_section_key() -> str | None:
        tab = input.main_tabs()
        return MAIN_TAB_TO_SECTION.get(tab)

    @reactive.calc
    def current_norm_method() -> str:
        tab = input.main_tabs()
        section = MAIN_TAB_TO_SECTION.get(tab)
        if section is not None:
            method = input[f"norm_method_{section}"]()
            if method:
                return _coerce_norm_method([method])
        if tab == TAB_ALL_IMPACTS:
            return _coerce_norm_method(
                [input.norm_method_radar() or _norm_method.get()]
            )
        return _norm_method.get()

    def _read_norm_method(radio_id: str) -> str:
        return _coerce_norm_method([input[radio_id]() or _norm_method.get()])

    def radar_norm_method() -> str:
        return _read_norm_method(NORM_METHOD_RADAR_ID)

    def bar_all_norm_method() -> str:
        return _read_norm_method(NORM_METHOD_BAR_ALL_ID)

    def section_norm_method(section_key: str) -> str:
        return _read_norm_method(f"norm_method_{section_key}")

    def _set_norm_method(method: str) -> None:
        method = _coerce_norm_method([method])
        _syncing_norm_method.set(True)
        try:
            _norm_method.set(method)
            for radio_id in _norm_radio_ids():
                ui.update_radio_buttons(radio_id, selected=method)
        finally:
            _syncing_norm_method.set(False)

    def _restore_full_impact_sidebar() -> None:
        data = current_data()
        all_choices = data.impact_choices
        backup = _impacts_before_section.get()
        if backup is not None:
            selected = _merge_selection(backup, all_choices)
        else:
            selected = _merge_selection(list(input.impact() or []), all_choices)
        _impacts_before_section.set(None)
        ui.update_checkbox_group(
            "impact",
            choices=all_choices,
            selected=selected,
        )
        ui.update_switch(
            "impact_all",
            value=bool(all_choices) and set(selected) == set(all_choices),
        )

    @reactive.effect
    @reactive.event(input.main_tabs)
    def _on_main_tab_changed() -> None:
        tab = input.main_tabs()
        section = MAIN_TAB_TO_SECTION.get(tab)
        prev = _section_active.get()
        if section != prev:
            if section is not None:
                if prev is None:
                    _impacts_before_section.set(list(input.impact() or []))
                _section_active.set(section)
                _enter_section_impact_sidebar(section)
            else:
                _section_active.set(None)
                if prev is not None:
                    _restore_full_impact_sidebar()

    @reactive.effect
    @reactive.event(input.impact)
    def _enforce_section_impact_lock() -> None:
        section = _section_active.get()
        if section is None or _locking_section_impacts.get():
            return
        allowed = set(_section_locked_impacts(section))
        current = set(input.impact() or [])
        if current <= allowed:
            return
        _show_section_impact_lock_notice()
        _update_section_impact_sidebar(
            section,
            selected=[i for i in (input.impact() or []) if i in allowed],
        )

    def _register_norm_method_sync(radio_id: str) -> None:
        @reactive.effect
        @reactive.event(input[radio_id])
        def _sync_norm_from_radio() -> None:
            if _syncing_norm_method.get():
                return
            method = input[radio_id]()
            if method and method != _norm_method.get():
                _set_norm_method(method)

        _sync_norm_from_radio.__name__ = f"_sync_norm_{radio_id}"

    for _radio_id in _norm_radio_ids():
        _register_norm_method_sync(_radio_id)

    def _sync_checkbox_groups(
        data: LcaData, *, previous: LcaData | None = None
    ) -> None:
        prev = previous if previous is not None else data
        old_fiber_cols = prev.fiber_cols
        old_impact_choices = prev.impact_choices
        current_fibers = list(input.fiber() or [])
        current_impacts = list(input.impact() or [])
        fiber_all = bool(old_fiber_cols) and (
            input.fiber_all()
            or set(current_fibers) >= set(old_fiber_cols)
        )
        impact_all = bool(old_impact_choices) and (
            input.impact_all()
            or set(current_impacts) >= set(old_impact_choices)
        )
        fiber_selected = _selection_after_catalog_reload(
            current_fibers,
            data.fiber_cols,
            all_was_selected=fiber_all,
        )
        impact_choices = data.impact_choices
        impact_selected = _selection_after_catalog_reload(
            current_impacts,
            impact_choices,
            all_was_selected=impact_all,
        )
        ui.update_checkbox_group(
            "fiber",
            choices=data.fiber_cols,
            selected=fiber_selected,
        )
        ui.update_checkbox_group(
            "impact",
            choices=impact_choices,
            selected=impact_selected,
        )
        ui.update_switch(
            "fiber_all",
            value=bool(data.fiber_cols) and set(fiber_selected) == set(data.fiber_cols),
        )
        ui.update_switch(
            "impact_all",
            value=bool(impact_choices)
            and set(impact_selected) == set(impact_choices),
        )
        section = _section_active.get()
        prev_applied = _applied_state.get()
        applied_fibers = [f for f in prev_applied.fibers if f in data.fiber_cols]
        if section is not None:
            allowed = set(_section_locked_impacts(section))
            applied_impacts = [i for i in prev_applied.impacts if i in allowed]
            picked = [i for i in impact_selected if i in allowed]
            _update_section_impact_sidebar(section, selected=picked)
        else:
            applied_impacts = [
                i for i in prev_applied.impacts if i in impact_choices
            ]
        _commit_applied_state(
            AppliedState(
                fibers=applied_fibers,
                impacts=applied_impacts,
            )
        )
        _live_fiber_cache.set(fiber_selected)
        if section is None:
            _live_impact_cache.set(impact_selected)

    @reactive.effect
    @reactive.event(input.fiber_all)
    def _apply_fiber_all() -> None:
        if _syncing_fiber_all_switch.get():
            return
        data = current_data()
        choices = data.fiber_cols
        current = list(input.fiber() or [])
        if input.fiber_all():
            if set(current) == set(choices):
                return
            selected = choices
        else:
            # Switch off after unchecking one item: keep partial selection.
            if set(current) != set(choices):
                return
            selected = []
        _applying_fiber_all.set(True)
        try:
            ui.update_checkbox_group("fiber", choices=choices, selected=selected)
        finally:
            _applying_fiber_all.set(False)

    @reactive.effect
    @reactive.event(input.impact_all)
    def _apply_impact_all() -> None:
        if _syncing_impact_all_switch.get():
            return
        section = _section_active.get()
        if section is not None:
            section_impacts = _section_locked_impacts(section)
            current = list(input.impact() or [])
            if input.impact_all():
                if set(current) == set(section_impacts):
                    return
                selected = section_impacts
            else:
                if set(current) != set(section_impacts):
                    return
                selected = []
            _applying_impact_all.set(True)
            try:
                _update_section_impact_sidebar(section, selected=selected)
            finally:
                _applying_impact_all.set(False)
            return
        data = current_data()
        choices = data.impact_choices
        current = list(input.impact() or [])
        if input.impact_all():
            if set(current) == set(choices):
                return
            selected = choices
        else:
            if set(current) != set(choices):
                return
            selected = []
        _applying_impact_all.set(True)
        try:
            ui.update_checkbox_group("impact", choices=choices, selected=selected)
        finally:
            _applying_impact_all.set(False)

    @reactive.effect
    @reactive.event(input.fiber)
    def _sync_fiber_all_switch() -> None:
        if _applying_fiber_all.get():
            return
        data = current_data()
        choices = data.fiber_cols
        want_on = bool(choices) and set(input.fiber() or []) == set(choices)
        if input.fiber_all() != want_on:
            _syncing_fiber_all_switch.set(True)
            try:
                ui.update_switch("fiber_all", value=want_on)
            finally:
                _syncing_fiber_all_switch.set(False)

    @reactive.effect
    @reactive.event(input.impact)
    def _sync_impact_all_switch() -> None:
        if _applying_impact_all.get() or _locking_section_impacts.get():
            return
        section = _section_active.get()
        if section is not None:
            section_impacts = _section_locked_impacts(section)
            want_on = (
                bool(section_impacts)
                and set(input.impact() or []) == set(section_impacts)
            )
            if input.impact_all() != want_on:
                _syncing_impact_all_switch.set(True)
                try:
                    ui.update_switch("impact_all", value=want_on)
                finally:
                    _syncing_impact_all_switch.set(False)
            return
        data = current_data()
        choices = data.impact_choices
        want_on = bool(choices) and set(input.impact() or []) == set(choices)
        if input.impact_all() != want_on:
            _syncing_impact_all_switch.set(True)
            try:
                ui.update_switch("impact_all", value=want_on)
            finally:
                _syncing_impact_all_switch.set(False)

    @reactive.effect
    def _reload_data_on_csv_change() -> None:
        reactive.invalidate_later(DATA_POLL_SECONDS)
        fresh = load_processed_data(DATA_FILE)
        previous = data_store.get()
        if not _data_catalog_changed(previous, fresh):
            return
        data_store.set(fresh)
        _sync_checkbox_groups(fresh, previous=previous)

    @output
    @render.ui
    def fiber_picker_summary():
        input.fiber()
        data = current_data()
        return _render_picker_summary(
            _live_fiber_cache.get(),
            data.fiber_cols,
            placeholder="Click to select fiber types…",
        )

    @output
    @render.ui
    def impact_picker_summary():
        input.impact()
        data = current_data()
        return _render_picker_summary(
            _live_impact_cache.get(),
            data.impact_choices,
            placeholder="Click to select environmental impacts…",
        )

    @output
    @render.ui
    def norm_formula_radar():
        return ui.div(
            _norm_formula_equation_ui(radar_norm_method()),
            class_="lca-norm-formula-block",
        )

    @output
    @render.ui
    def norm_formula_bar_all():
        return ui.div(
            _norm_formula_equation_ui(bar_all_norm_method()),
            class_="lca-norm-formula-block",
        )

    @output(id="data_tab_raw", suspend_when_hidden=True)
    @render.data_frame
    def data_tab_raw():
        return comparison_table_raw_df()

    @output(id="data_tab_normalized", suspend_when_hidden=True)
    @render.data_frame
    def data_tab_normalized():
        return comparison_table_normalized_df()

    _register_table_csv_download_handler(
        output,
        render,
        table_id="data_tab_raw",
        df_fn=comparison_table_raw_df,
        basename_fn=lambda: "lca-comparison-table-raw",
    )
    _register_table_csv_download_handler(
        output,
        render,
        table_id="data_tab_normalized",
        df_fn=comparison_table_normalized_df,
        basename_fn=lambda: "lca-comparison-table-normalized",
    )

    @reactive.calc
    def all_impacts_figure() -> go.Figure:
        state = _applied_state.get()
        data = current_data()
        chart_id = input.chart_kind_all()
        if chart_id in ("bar_vertical", "bar_horizontal"):
            norm_method = bar_all_norm_method()
        elif chart_id == "radar":
            norm_method = radar_norm_method()
        else:
            norm_method = NORM_METHOD_RAW
        return _build_chart_figure(
            chart_id,
            list(state.impacts),
            list(state.fibers),
            data.df,
            data.impact_col,
            norm_method=norm_method,
        )

    @output(id="all_impacts_plot", suspend_when_hidden=True)
    @render_plotly
    def all_impacts_plot():
        return _plotly_widget(all_impacts_figure())

    _register_chart_download_handlers(
        output,
        render,
        plot_id="all_impacts_plot",
        figure_fn=all_impacts_figure,
        basename_fn=lambda: "lca-all-impacts",
    )

    @output(id="all_impacts_toolbar", suspend_when_hidden=True)
    @render.ui
    def all_impacts_toolbar() -> ui.Tag:
        chart_id = input.chart_kind_all()
        items: list[ui.Tag] = []
        if chart_id == "heatmap":
            items.append(ui.output_ui("all_impacts_values_preview"))
            items.append(_heatmap_guide_panel_ui())
        elif chart_id == "radar":
            items.append(
                _normalization_hover_panel_ui(
                    method_input_id="norm_method_radar",
                    formula_output_id="norm_formula_radar",
                )
            )
            items.append(ui.output_ui("all_impacts_values_preview"))
            items.append(_radar_guide_panel_ui())
        elif chart_id == "bar_vertical":
            items.append(
                _normalization_hover_panel_ui(
                    method_input_id="norm_method_bar_all",
                    formula_output_id="norm_formula_bar_all",
                )
            )
            items.append(ui.output_ui("all_impacts_values_preview"))
            items.append(_all_impacts_vertical_bar_guide_panel_ui())
        elif chart_id == "bar_horizontal":
            items.append(
                _normalization_hover_panel_ui(
                    method_input_id="norm_method_bar_all",
                    formula_output_id="norm_formula_bar_all",
                )
            )
            items.append(ui.output_ui("all_impacts_values_preview"))
            items.append(_all_impacts_horizontal_bar_guide_panel_ui())
        if not items:
            return ui.div()
        return ui.div(
            *items,
            class_="d-flex flex-wrap align-items-center gap-3 px-2 pt-2 pb-1",
        )

    @output(id="all_impacts_values_preview", suspend_when_hidden=True)
    @render.ui
    def all_impacts_values_preview():
        state = _applied_state.get()
        data = current_data()
        chart_id = input.chart_kind_all()
        if chart_id == "heatmap":
            norm_method = NORM_METHOD_RAW
        elif chart_id == "radar":
            norm_method = radar_norm_method()
        elif chart_id in ("bar_vertical", "bar_horizontal"):
            norm_method = bar_all_norm_method()
        else:
            norm_method = NORM_METHOD_RAW
        return _all_impacts_values_preview_tag(
            data,
            list(state.impacts),
            list(state.fibers),
            norm_method=norm_method,
        )

    def _register_impact_section_outputs(section_key: str) -> None:
        @reactive.calc
        def section_figure() -> go.Figure:
            state = _applied_state.get()
            data = current_data()
            section_impacts = _section_chart_impacts(section_key)
            return build_bar_figure(
                section_impacts,
                list(state.fibers),
                data.df,
                data.impact_col,
                norm_method=section_norm_method(section_key),
            )

        section_figure.__name__ = f"section_figure_{section_key}"

        @output(id=f"section_note_{section_key}")
        @render.ui
        def section_note() -> ui.Tag:
            data = current_data()
            return _impact_section_note_ui(section_key, data.impact_choices)

        @output(id=f"section_toolbar_{section_key}", suspend_when_hidden=True)
        @render.ui
        def section_toolbar() -> ui.Tag:
            return ui.div(
                _normalization_hover_panel_ui(
                    method_input_id=f"norm_method_{section_key}",
                    formula_output_id=f"norm_formula_{section_key}",
                ),
                _bar_chart_guide_panel_ui(),
                class_="d-flex flex-wrap align-items-center gap-3 px-2 pt-2 pb-1",
            )

        @output(id=f"norm_formula_{section_key}")
        @render.ui
        def section_norm_formula() -> ui.Tag:
            return ui.div(
                _norm_formula_equation_ui(section_norm_method(section_key)),
                class_="lca-norm-formula-block",
            )

        @output(id=f"section_plot_{section_key}", suspend_when_hidden=True)
        @render_plotly
        def section_plot() -> go.FigureWidget:
            return _plotly_widget(section_figure())

        _register_chart_download_handlers(
            output,
            render,
            plot_id=f"section_plot_{section_key}",
            figure_fn=section_figure,
            basename_fn=lambda sk=section_key: f"lca-{sk}-chart",
        )

    for _section_key in IMPACT_SECTIONS:
        _register_impact_section_outputs(_section_key)

    _HOME_NAV_BUTTONS: dict[str, str] = {
        "nav_home_title": TAB_HOME,
        "nav_method_sidebar": TAB_METHOD,
        "home_nav_comparison_table": TAB_COMPARISON,
        "home_nav_all_impacts": TAB_ALL_IMPACTS,
        "home_nav_method": TAB_METHOD,
        "home_nav_team": TAB_TEAM,
        **{str(card["button_id"]): str(card["tab"]) for card in HOME_SECTION_CARDS},
    }

    def _register_home_nav_button(button_id: str, tab_value: str) -> None:
        @reactive.effect
        @reactive.event(input[button_id])
        def _go_to_tab() -> None:
            ui.update_navs("main_tabs", selected=tab_value)

        _go_to_tab.__name__ = f"_home_nav_{tab_value}"

    for _btn_id, _tab_value in _HOME_NAV_BUTTONS.items():
        _register_home_nav_button(_btn_id, _tab_value)




app = App(app_ui, server)

"""rsconnect deploy shiny ."""
"""rsconnect deploy shiny ."""