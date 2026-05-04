from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import matplotlib.cm as mcm
import seaborn as sns

from cgnn.utils.codebook import TITLE_CBSA_MAP
from cgnn.utils.utils import get_cbsa_list

import geopandas as gpd

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader


def _set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    return fig_width_in, fig_height_in


def plot_explainable_subgraph(
    explain_edges_df,
    target_date,
    hosp_df=None,          # DataFrame with columns cbsa_col and hosp_col
    hosp_col='total_adult_patients_hospitalized_confirmed_covid_7_day_sum',
    cbsa_col='cbsa',       # column name in hosp_df containing CBSA codes
    date_col=None,         # column name in hosp_df for date; if set, filters to target_date
    land_color='#F0F0F0',
    origin_cbsas=None,     # list of CBSA codes; if set, plot edges leaving these CBSAs
    dest_cbsas=None,       # list of CBSA codes; if set, plot edges arriving at these CBSAs
    shade_cbsas=None,      # list of CBSA codes; if set, only shade these CBSAs
    importance_masks=None,     # DataFrame with date/cbsa_orig/cbsa_dest/importance columns
    importance_threshold=None, # float in [0,1]; edges below this are dropped, alpha rescaled
    curved_edges=False,        # if True, draw arrows as arcs; if False, draw straight arrows
    show_timeseries=True,      # if False, suppress the time series panel even when data is available
    ax=None,                   # if set, render map into this cartopy GeoAxes (no fig created)
    save_path=None,            # if set, save figure to this path
    suppress_legend=False,     # if True, skip drawing the per-map edge legend
    hosp_cbar_ax=None,         # if set, draw hosp colorbar into this axis (used by grid)
):
    """
    Plot the explainable subgraph for a given date, with CBSA polygons shaded
    by hospitalization count and edges drawn as arrows between centroids.

    Args:
        explain_edges_df: DataFrame with columns 'date', 'cbsa_orig', 'cbsa_dest'
        target_date: date to plot (string 'YYYY-MM-DD' or datetime)
        hosp_df: optional DataFrame with cbsa_col and hosp_col for polygon shading
        hosp_col: column name in hosp_df to use for shading
        cbsa_col: column name in hosp_df containing CBSA codes (default 'cbsa')
        date_col: column name in hosp_df to filter by target_date (e.g. 'collection_week');
            if None, hosp_df must already be filtered to a single date
        land_color: background land color
        origin_cbsas: optional list of CBSA codes; if provided, edges where cbsa_orig
            is in this list are plotted (outgoing edges, drawn in blue)
        dest_cbsas: optional list of CBSA codes; if provided, edges where cbsa_dest
            is in this list are plotted (incoming edges, drawn in orange); can be used
            independently or together with origin_cbsas
        shade_cbsas: optional list of CBSA codes; if provided, only these CBSAs
            are shaded by hospitalization count
        importance_masks: optional DataFrame with columns 'date', 'cbsa_orig',
            'cbsa_dest', 'importance' (float in [0, 1]); used to set edge alpha
        curved_edges: if True, draw arrows as arcs (rad=0.2); default False (straight)

    Returns:
        matplotlib.figure.Figure
    """
    _repo_root = Path(__file__).parent.parent.parent
    cbsa_gdf = gpd.read_file(
        _repo_root / 'assets' / 'cb_2018_us_cbsa_500k' / 'cb_2018_us_cbsa_500k.shp'
    )
    states_gdf = gpd.read_file(
        _repo_root / 'assets' / 'cb_2018_us_state_500k' / 'cb_2018_us_state_500k.shp'
    )
    raw_advan_mobility = pd.read_csv(
        _repo_root / 'data' / 'raw' / 'mobility' / 'advan_plus' / 'all_advan_plus.csv',
        dtype={'cbsa_orig': str, 'cbsa_dest': str},
    )

    _cbsa_info_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'list1_2023.csv'
    _cbsa_info = pd.read_csv(_cbsa_info_path, usecols=['CBSA Code', 'State Name'])

    _pop_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'cbsa-est2024-alldata.csv'
    _pop_raw = pd.read_csv(
        _pop_path,
        usecols=['CBSA', 'MDIV', 'STCOU', 'POPESTIMATE2024'],
        dtype={'CBSA': str},
        encoding='ISO-8859-1',
    )
    # Keep only CBSA-level rows — county and metro-division sub-rows share the same
    # CBSA code but have non-null STCOU/MDIV; joining on those inflates the rate
    pop_df = _pop_raw[_pop_raw['STCOU'].isna() & _pop_raw['MDIV'].isna()][
        ['CBSA', 'POPESTIMATE2024']
    ]
    _non_conus = set(
        _cbsa_info[_cbsa_info['State Name'].isin({'Alaska', 'Hawaii', 'Puerto Rico'})]
        ['CBSA Code'].astype(str)
    )
    explain_edges_df = explain_edges_df[
        ~explain_edges_df['cbsa_orig'].astype(str).isin(_non_conus) &
        ~explain_edges_df['cbsa_dest'].astype(str).isin(_non_conus)
    ]

    target_date = pd.to_datetime(target_date)

    # --- Compute centroids for edge endpoints ---
    # Project to Albers Equal Area (EPSG:5070) for accurate centroids, then back to lon/lat
    cbsa_projected = cbsa_gdf[['GEOID', 'geometry']].to_crs(epsg=5070)
    cbsa_centroids = cbsa_projected.copy()
    cbsa_centroids['geometry'] = cbsa_projected.geometry.centroid
    cbsa_centroids = cbsa_centroids.to_crs(epsg=4326)
    centroids = cbsa_centroids[['GEOID']].copy()
    centroids['LON'] = cbsa_centroids.geometry.x
    centroids['LAT'] = cbsa_centroids.geometry.y

    # --- Merge centroids into edge DataFrame ---
    _origin_set = {str(c) for c in origin_cbsas} if origin_cbsas is not None else set()
    _dest_set = {str(c) for c in dest_cbsas} if dest_cbsas is not None else set()

    edges = explain_edges_df[explain_edges_df['date'] == target_date]
    if _origin_set or _dest_set:
        edges = edges[
            edges['cbsa_orig'].astype(str).isin(_origin_set) |
            edges['cbsa_dest'].astype(str).isin(_dest_set)
        ]
    subset_df = (
        edges
        .copy()
        .merge(
            centroids.rename(columns={'GEOID': 'cbsa_orig', 'LON': 'LON_orig', 'LAT': 'LAT_orig'}),
            on='cbsa_orig', how='left',
        )
        .merge(
            centroids.rename(columns={'GEOID': 'cbsa_dest', 'LON': 'LON_dest', 'LAT': 'LAT_dest'}),
            on='cbsa_dest', how='left',
        )
        .dropna(subset=['LAT_orig', 'LON_orig', 'LAT_dest', 'LON_dest'])
    )

    inter_edges = subset_df[subset_df['cbsa_orig'] != subset_df['cbsa_dest']].copy()
    self_loops  = subset_df[subset_df['cbsa_orig'] == subset_df['cbsa_dest']]

    # Tag direction: outgoing (orig in origin_set) takes priority when both match
    inter_edges['is_incoming'] = (
        ~inter_edges['cbsa_orig'].astype(str).isin(_origin_set) &
        inter_edges['cbsa_dest'].astype(str).isin(_dest_set)
    )

    if raw_advan_mobility is not None:
        mob = raw_advan_mobility[
            pd.to_datetime(raw_advan_mobility['date_range_start']) == target_date
        ][['cbsa_orig', 'cbsa_dest', 'visitor_home_aggregation']]
        inter_edges = inter_edges.merge(mob, on=['cbsa_orig', 'cbsa_dest'], how='left')
        log_mob = np.log1p(inter_edges['visitor_home_aggregation'].fillna(0))
        _mob_max = log_mob.max()
        inter_edges['edge_color_val'] = (log_mob / _mob_max).clip(0, 1) if _mob_max > 0 else 0.5
    elif importance_masks is not None:
        imp = (
            importance_masks[importance_masks['date'] == target_date]
            [['cbsa_orig', 'cbsa_dest', 'importance']]
        )
        inter_edges = inter_edges.merge(imp, on=['cbsa_orig', 'cbsa_dest'], how='left')
        inter_edges['importance'] = inter_edges['importance'].clip(0, 1).fillna(0.0)
        if importance_threshold is not None:
            inter_edges = inter_edges[inter_edges['importance'] >= importance_threshold]
            scale = 1.0 - importance_threshold
            inter_edges['importance'] = (
                (inter_edges['importance'] - importance_threshold) / scale
            )
        inter_edges['edge_color_val'] = inter_edges['importance']
    else:
        inter_edges['edge_color_val'] = 0.7

    # --- Figure Setup ---
    platecarree = ccrs.PlateCarree()
    if ax is None:
        show_timeseries = (show_timeseries and (_origin_set or _dest_set) and hosp_df is not None and date_col is not None)
        if show_timeseries:
            fig = plt.figure(figsize=(12, 11), dpi=300)
            gs = fig.add_gridspec(2, 1, height_ratios=[5, 1])
            ax = fig.add_subplot(gs[0], projection=platecarree)
            ax_ts = fig.add_subplot(gs[1])
        else:
            fig = plt.figure(figsize=(12, 8), dpi=300)
            ax = fig.add_subplot(1, 1, 1, projection=platecarree)
    else:
        fig = None
        show_timeseries = False
    ax.set_extent([-125, -66, 24, 50], platecarree)
    ax.spines['geo'].set_visible(False)

    # --- Geographic Features (US only) ---
    ax.set_facecolor('white')  # ocean/non-US background
    if states_gdf is not None:
        ax.add_geometries(states_gdf.geometry, crs=platecarree, facecolor=land_color,
                          edgecolor='gray', linewidth=0.4, zorder=5)
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black', zorder=6)

    # --- CBSA outlines (CONUS only) ---
    conus_mask = ~cbsa_gdf['GEOID'].isin(_non_conus)
    outline_gdf = cbsa_gdf[conus_mask] if shade_cbsas is None else cbsa_gdf[conus_mask & cbsa_gdf['GEOID'].isin(shade_cbsas)]
    ax.add_geometries(
        outline_gdf.geometry,
        crs=platecarree,
        facecolor='none',
        edgecolor='black',
        linewidth=0.2,
        alpha=0.4,
        zorder=7,
    )

    # --- CBSA polygon shading by hospitalization rate ---
    if hosp_df is not None:
        _hosp = hosp_df
        if date_col is not None:
            _hosp = _hosp[pd.to_datetime(_hosp[date_col]) == target_date]
        _hosp_merge = _hosp[[cbsa_col, hosp_col]].copy()
        _hosp_merge[cbsa_col] = _hosp_merge[cbsa_col].astype(str)
        merged_gdf = cbsa_gdf.merge(
            _hosp_merge.rename(columns={cbsa_col: 'GEOID'}),
            on='GEOID', how='left',
        ).merge(
            pop_df.rename(columns={'CBSA': 'GEOID'}),
            on='GEOID', how='left',
        )
        merged_gdf['hosp_rate'] = (
            merged_gdf[hosp_col] / merged_gdf['POPESTIMATE2024'] * 100_000
        )
        present = merged_gdf[merged_gdf['hosp_rate'].notna()]
        if shade_cbsas is not None:
            present = present[present['GEOID'].isin(shade_cbsas)]
        print(present['hosp_rate'].min(), present['hosp_rate'].max())
        norm = mcolors.Normalize(vmin=present['hosp_rate'].min(), vmax=present['hosp_rate'].max())
        cmap = mcm.YlOrRd

        for _, row in present.iterrows():
            ax.add_geometries(
                [row.geometry],
                crs=platecarree,
                facecolor=cmap(norm(row['hosp_rate'])),
                edgecolor='none',
                alpha=0.85,
                zorder=6,
            )

        if fig is not None or hosp_cbar_ax is not None:
            sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            if hosp_cbar_ax is not None:
                cbar = hosp_cbar_ax.figure.colorbar(sm, cax=hosp_cbar_ax, orientation='horizontal')
            else:
                cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                                    fraction=0.02, pad=0.04, shrink=0.6, aspect=40)
            cbar.set_label('Hospitalizations per 100k Population', fontsize=9)
            cbar.ax.tick_params(labelsize=8)

    _out_cmap = mcm.Blues
    _in_cmap  = mcm.Oranges
    # Sort ascending so high-mobility edges are drawn on top
    for _, row in inter_edges.sort_values('edge_color_val').iterrows():
        cv = float(row['edge_color_val'])
        cv_scaled = cv ** 3
        cmap = _in_cmap if row['is_incoming'] else _out_cmap
        r, g, b, _ = cmap(0.35 + 0.65 * cv_scaled)
        alpha = float(np.clip(cv_scaled * 1.5, 0.05, 1.0))
        ax.annotate(
            '',
            xy=(row['LON_dest'], row['LAT_dest']),
            xycoords=platecarree,
            xytext=(row['LON_orig'], row['LAT_orig']),
            textcoords=platecarree,
            arrowprops=dict(
                arrowstyle='->',
                color=(r, g, b, alpha),
                lw=0.8,
                shrinkA=0, shrinkB=0,
                connectionstyle=f'arc3,rad={0.15}' if curved_edges else f'arc3,rad=0.0',
            ),
            zorder=7,
        )

    legend_title = 'log(Visitor Flow)\n(normalized)' if raw_advan_mobility is not None else 'Edge Importance'
    lo = importance_threshold if (importance_threshold is not None and raw_advan_mobility is None) else 0.0
    legend_levels = [lo + (1.0 - lo) * t for t in [0.0, 1/3, 2/3, 1.0]]
    legend_handles = []
    if _origin_set:
        legend_handles += [
            Line2D([0], [0], color=_out_cmap(0.35 + 0.65 * ((lvl - lo) / (1.0 - lo) if lo < 1.0 else 1.0) ** 2),
                   lw=1.5, label=f'Out {lvl:.2f}')
            for lvl in legend_levels
        ]
    if _dest_set:
        legend_handles += [
            Line2D([0], [0], color=_in_cmap(0.35 + 0.65 * ((lvl - lo) / (1.0 - lo) if lo < 1.0 else 1.0) ** 2),
                   lw=1.5, label=f'In {lvl:.2f}')
            for lvl in legend_levels
        ]
    if not legend_handles:
        legend_handles = [
            Line2D([0], [0], color=_out_cmap(0.35 + 0.65 * ((lvl - lo) / (1.0 - lo) if lo < 1.0 else 1.0) ** 2),
                   lw=1.5, label=f'{lvl:.2f}')
            for lvl in legend_levels
        ]
    if not suppress_legend:
        ax.legend(
            handles=legend_handles,
            title=legend_title,
            title_fontsize=8,
            fontsize=8,
            loc='lower left',
            framealpha=0.8,
            ncol=2,
        )

    _thresh_str = (
        f", importance threshold={importance_threshold}"
        if importance_threshold is not None else ""
    )
    _n_out = int((~inter_edges['is_incoming']).sum())
    _n_in  = int(inter_edges['is_incoming'].sum())
    _edge_str = ', '.join(filter(None, [
        f'{_n_out} outgoing' if _origin_set else '',
        f'{_n_in} incoming' if _dest_set else '',
    ]))
    _title = (
        f"Explainable Subgraph — {target_date.strftime('%Y-%m-%d')}\n"
        f"({_edge_str} edges, {len(self_loops)} self-loops{_thresh_str})"
    )
    if _origin_set:
        _origin_titles = ', '.join(TITLE_CBSA_MAP.get(c, c) for c in _origin_set)
        _title += f"\nOrigin: {_origin_titles}"
    if _dest_set:
        _dest_titles = ', '.join(TITLE_CBSA_MAP.get(c, c) for c in _dest_set)
        _title += f"\nDest: {_dest_titles}"
    ax.set_title(_title, fontsize=10, pad=10)

    if show_timeseries:
        _ts = hosp_df.copy()
        _ts_cbsas = _origin_set | _dest_set
        _ts = _ts[_ts[cbsa_col].astype(str).isin(_ts_cbsas)]
        _ts[date_col] = pd.to_datetime(_ts[date_col])
        _ts[cbsa_col] = _ts[cbsa_col].astype(str)
        _ts = _ts.merge(
            pop_df.rename(columns={'CBSA': cbsa_col}),
            on=cbsa_col, how='left',
        )
        _ts['hosp_rate'] = _ts[hosp_col] / _ts['POPESTIMATE2024'] * 100_000
        _ts_ymax = _ts['hosp_rate'].max()
        _ts_xmin, _ts_xmax = _ts[date_col].min(), _ts[date_col].max()
        _ts = _ts[_ts[date_col] <= target_date].sort_values(date_col)
        hosp_lines = []
        for cbsa, grp in _ts.groupby(cbsa_col):
            label = TITLE_CBSA_MAP.get(str(cbsa), str(cbsa)) if len(_ts_cbsas) > 1 else "Hospitalization rate"
            ln, = ax_ts.plot(grp[date_col], grp['hosp_rate'], linewidth=1.2, label=label)
            hosp_lines.append(ln)
        ax_ts.axvline(target_date, color='#c0392b', linestyle='--', linewidth=0.9, zorder=3)
        ax_ts.set_xlim(_ts_xmin, _ts_xmax)
        ax_ts.set_ylim(0, _ts_ymax * 1.05)
        ax_ts.set_ylabel('Hospitalizations per 100k', fontsize=9)
        ax_ts.tick_params(axis='y', labelsize=8)
        ax_ts.tick_params(axis='x', labelsize=8, rotation=45)
        ax_ts.spines['top'].set_visible(False)

        # --- Secondary y-axis: outgoing and/or incoming edge counts ---
        _non_self = explain_edges_df[
            explain_edges_df['cbsa_orig'] != explain_edges_df['cbsa_dest']
        ]

        def _edge_counts(df, col, cbsa_set):
            return (
                df[df[col].astype(str).isin(cbsa_set)]
                .groupby('date').size()
                .reset_index(name='edge_count')
                .assign(date=lambda d: pd.to_datetime(d['date']))
            )

        ax_ec = ax_ts.twinx()
        ec_lines = []
        _ec_global_max = 1

        if _origin_set:
            _ec_out_all = _edge_counts(_non_self, 'cbsa_orig', _origin_set)
            _ec_global_max = max(_ec_global_max, _ec_out_all['edge_count'].max() if len(_ec_out_all) > 0 else 1)
            _ec_out = _ec_out_all[_ec_out_all['date'] <= target_date].sort_values('date')
            ln_out, = ax_ec.step(
                _ec_out['date'], _ec_out['edge_count'],
                where='mid', color=_out_cmap(0.7), linewidth=1.0,
                linestyle=':', label='Outgoing edges', zorder=2,
            )
            ec_lines.append(ln_out)

        if _dest_set:
            _ec_in_all = _edge_counts(_non_self, 'cbsa_dest', _dest_set)
            _ec_global_max = max(_ec_global_max, _ec_in_all['edge_count'].max() if len(_ec_in_all) > 0 else 1)
            _ec_in = _ec_in_all[_ec_in_all['date'] <= target_date].sort_values('date')
            ln_in, = ax_ec.step(
                _ec_in['date'], _ec_in['edge_count'],
                where='mid', color=_in_cmap(0.7), linewidth=1.0,
                linestyle=':', label='Incoming edges', zorder=2,
            )
            ec_lines.append(ln_in)

        ax_ec.set_ylim(0, _ec_global_max * 1.3)
        ax_ec.set_ylabel('# Subgraph Edges', fontsize=9)
        ax_ec.tick_params(axis='y', labelsize=8)
        ax_ec.spines['top'].set_visible(False)

        # Combined legend: hosp lines + edge counts + target date marker
        target_line = Line2D(
            [0], [0], color='#c0392b', linestyle='--', linewidth=0.9, label='Target date'
        )
        ax_ts.legend(
            handles=hosp_lines + ec_lines + [target_line],
            fontsize=8, loc='upper left', framealpha=0.85,
            edgecolor='#cccccc', borderpad=0.6,
        )

    if fig is not None:
        if show_timeseries:
            fig.subplots_adjust(hspace=-0.3)
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    return fig


def plot_bump_chart(
    explain_edges_count_df,
    top_n=10,
    min_years=2,   # drop CBSAs that appear in top_n fewer than this many years
    rank_cap=10,   # unused; kept for backward compatibility
    figsize=None,  # defaults to scaling with top_n
):
    """
    Bump chart of top-N CBSA edge origins per year.

    Lines are drawn only within the top_n band. A downward triangle (▼) marks
    the last year a CBSA is in top_n before dropping out; an upward triangle (▲)
    marks re-entry.

    Args:
        explain_edges_count_df: DataFrame with columns 'year', 'cbsa_orig',
            'rank', 'cbsa_title'
        top_n: number of top CBSAs to highlight per year
        min_years: minimum years a CBSA must appear in top_n to be included
        rank_cap: unused; kept for backward compatibility
        figsize: figure size tuple

    Returns:
        matplotlib.figure.Figure
    """
    top_cbsa_per_year = explain_edges_count_df[
        explain_edges_count_df['rank'] <= top_n
    ].copy()

    # Filter: only CBSAs in top_n for at least min_years years
    years_in_top = top_cbsa_per_year.groupby('cbsa_orig')['year'].nunique()
    persistent_cbsas = set(years_in_top[years_in_top >= min_years].index)
    top_cbsa_per_year = top_cbsa_per_year[top_cbsa_per_year['cbsa_orig'].isin(persistent_cbsas)]

    if figsize is None:
        figsize = (14, top_n * 0.6 + 2)

    cbsa_list_for_plot = top_cbsa_per_year['cbsa_orig'].drop_duplicates().tolist()
    tab20 = plt.get_cmap('tab20').colors
    color_map = {cbsa: tab20[i % 20] for i, cbsa in enumerate(cbsa_list_for_plot)}

    all_years = sorted(explain_edges_count_df['year'].unique())
    first_year, last_year = all_years[0], all_years[-1]
    fig, ax = plt.subplots(figsize=figsize)

    for cbsa, group in top_cbsa_per_year.groupby('cbsa_orig'):
        group = group.sort_values('year').reset_index(drop=True)
        color = color_map[cbsa]
        years_list = group['year'].tolist()
        ranks_list = group['rank'].tolist()

        # Solid segments only between consecutive years within top_n
        for i in range(len(years_list) - 1):
            if years_list[i + 1] - years_list[i] == 1:
                ax.plot([years_list[i], years_list[i + 1]],
                        [ranks_list[i], ranks_list[i + 1]],
                        color=color, linewidth=2, zorder=2)

        # Label at every year the CBSA appears in top_n
        for _, row in group.iterrows():
            label = f"{int(row['rank'])}. {row['cbsa_title']}"
            ax.text(
                x=row['year'],
                y=row['rank'],
                s=label,
                va='center', ha='center',
                fontsize=8, color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, linewidth=0.8, alpha=0.85),
                zorder=4,
            )

    ax.set_xlabel("Year")
    ax.set_ylabel("Rank (1 = most-edge-origin CBSA)")
    ax.set_title(f"Bump Chart of Top {top_n} CBSA Edge Origins Per Year")
    ax.invert_yaxis()
    ax.set_xticks(all_years)
    ax.set_yticks(range(1, top_n + 1))
    ax.set_ylim(top_n + 0.5, 0.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig


def plot_subgraph_heatmap_grid(
    dates,
    explain_edges_df,
    importance_masks,
    hosp_df=None,
    hosp_col='total_adult_patients_hospitalized_confirmed_covid_7_day_sum',
    cbsa_col='CBSA',
    date_col=None,
    origin_cbsas=None,
    dest_cbsas=None,
    shade_cbsas=None,
    importance_threshold=0.5,
    curved_edges=False,
    mask_diagonal=True,
    height_ratios=(2, 3),
    figsize=None,
    save_path=None,
):
    """
    n-row combined figure: each row = one date, with an explainable subgraph
    map on the left and side-by-side mobility/subgraph heatmaps on the right.

    Args:
        dates: list of date strings ('YYYY-MM-DD')
        explain_edges_df: passed to plot_explainable_subgraph
        importance_masks: passed to both plot functions
        hosp_df, hosp_col, cbsa_col, date_col: passed to plot_explainable_subgraph
        origin_cbsas, dest_cbsas, shade_cbsas: passed to plot_explainable_subgraph
        importance_threshold: passed to both plot functions
        curved_edges: passed to plot_explainable_subgraph
        mask_diagonal: passed to plot_mobility_importance_heatmap
        height_ratios: unused (kept for backward compatibility)
        figsize: overall figure size; defaults to (18, 5*n_dates)
        save_path: if set, save figure here

    Returns:
        matplotlib.figure.Figure
    """
    n = len(dates)
    assert n >= 1, "dates must be a non-empty list"

    platecarree = ccrs.PlateCarree()

    if figsize is None:
        figsize = (18, 5 * n)

    # GridSpec: (n+1) rows × 3 cols — map | mob heatmap | sub heatmap
    #           bottom row: hosp cbar | heatmap cbar (spanning cols 1-2)
    fig = plt.figure(figsize=figsize, layout='constrained')
    gs = GridSpec(
        n + 1, 3, figure=fig,
        width_ratios=[3, 1, 1],
        height_ratios=[1] * n + [0.03],
    )

    # Colorbar axes in the bottom row
    hosp_cbar_ax = fig.add_subplot(gs[n, 0])
    cbar_ax      = fig.add_subplot(gs[n, 1:])

    _ax_maps, _ax_mobs, _ax_subs = [], [], []

    for i, date_str in enumerate(dates):
        # ── Map panel ────────────────────────────────────────────────────
        ax_map = fig.add_subplot(gs[i, 0], projection=platecarree)
        plot_explainable_subgraph(
            explain_edges_df,
            date_str,
            hosp_df=hosp_df,
            hosp_col=hosp_col,
            cbsa_col=cbsa_col,
            date_col=date_col,
            origin_cbsas=origin_cbsas,
            dest_cbsas=dest_cbsas,
            shade_cbsas=shade_cbsas,
            importance_masks=importance_masks,
            importance_threshold=importance_threshold,
            curved_edges=curved_edges,
            show_timeseries=False,
            ax=ax_map,
            suppress_legend=True,
            hosp_cbar_ax=hosp_cbar_ax if i == 0 else None,
        )
        ax_map.set_title(pd.to_datetime(date_str).strftime('%Y-%m-%d'),
                         fontsize=10, fontweight='bold', pad=6)
        _ax_maps.append(ax_map)

        # ── Heatmap panels ────────────────────────────────────────────────
        ax_mob = fig.add_subplot(gs[i, 1])
        ax_sub = fig.add_subplot(gs[i, 2])
        plot_mobility_importance_heatmap(
            date_str,
            importance_masks=importance_masks,
            importance_threshold=importance_threshold,
            mask_diagonal=mask_diagonal,
            axes=(ax_mob, ax_sub, cbar_ax),
            draw_colorbar=(i == 0),
            show_y_labels=True,
        )
        _ax_mobs.append(ax_mob)
        _ax_subs.append(ax_sub)

    # ── Shared edge legend on the first map ──────────────────────────────
    _out_cmap = mcm.Blues
    _in_cmap  = mcm.Oranges
    _legend_title  = 'log(Visitor Flow)\n(normalized)'
    _legend_levels = [0.0, 1/3, 2/3, 1.0]
    _legend_handles = []
    if origin_cbsas:
        _legend_handles += [
            Line2D([0], [0], color=_out_cmap(0.35 + 0.65 * lvl ** 2),
                   lw=1.5, label=f'Out {lvl:.2f}')
            for lvl in _legend_levels
        ]
    if dest_cbsas:
        _legend_handles += [
            Line2D([0], [0], color=_in_cmap(0.35 + 0.65 * lvl ** 2),
                   lw=1.5, label=f'In {lvl:.2f}')
            for lvl in _legend_levels
        ]
    if not _legend_handles:
        _legend_handles = [
            Line2D([0], [0], color=_out_cmap(0.35 + 0.65 * lvl ** 2),
                   lw=1.5, label=f'{lvl:.2f}')
            for lvl in _legend_levels
        ]
    _ax_maps[0].legend(
        handles=_legend_handles,
        title=_legend_title,
        title_fontsize=8,
        fontsize=8,
        loc='lower left',
        framealpha=0.8,
        ncol=2,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        import os
        base, ext = os.path.splitext(save_path)
        low_mem_path = f"{base}_low_mem.png"
        fig.savefig(low_mem_path, dpi=50, bbox_inches='tight', pad_inches=0.05)
    return fig


def plot_mobility_importance_heatmap(
    target_date,
    importance_masks=None,     # DataFrame with date/cbsa_orig/cbsa_dest/importance
    importance_threshold=None, # drop subgraph edges below this importance
    mask_diagonal=True,        # mask self-loops (diagonal) so off-diagonal flow is visible
    axes=None,                 # if set, tuple of (ax_mob, ax_sub, cbar_ax); no fig created
    draw_colorbar=True,        # if False, skip drawing the colorbar (e.g. when sharing one)
    show_y_labels=True,        # if False, hide HHS region labels on the y-axis
    save_path=None,
):
    """
    Side-by-side heatmaps of (1) raw mobility and (2) GNNExplainer subgraph importance
    for a single date. The CBSA universe and axis ordering are fixed across all dates
    using get_cbsa_list(), sorted west to east by centroid longitude, so the matrix
    layout is consistent when animating or comparing across weeks.

    Missing mobility flows are filled with 0. Missing importance values (edges not
    selected by the explainer) are also filled with 0.

    Args:
        target_date: date string 'YYYY-MM-DD' or datetime
        importance_masks: optional DataFrame with columns date, cbsa_orig, cbsa_dest,
            importance (float in [0, 1])
        importance_threshold: if set, edges with importance below this are set to 0
        mask_diagonal: if True, mask the diagonal (self-loops) in both panels
        save_path: if set, save figure to this path

    Returns:
        matplotlib.figure.Figure
    """
    target_date = pd.to_datetime(target_date)

    advan_df = pd.read_csv(
        Path(__file__).parent.parent.parent / 'data' / 'raw' / 'mobility' / 'advan_plus' / 'all_advan_plus.csv',
        dtype={'cbsa_orig': str, 'cbsa_dest': str},
    )

    # --- Load CBSA centroids for geographic sorting (west → east) ---
    _gaz_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / '2024_Gaz_cbsa_national.csv'
    _gaz = pd.read_csv(_gaz_path, usecols=['GEOID', 'INTPTLONG'], dtype={'GEOID': str})
    _lon_map = _gaz.set_index('GEOID')['INTPTLONG'].astype(float).to_dict()

    def _geo_sort_key(code):
        return _lon_map.get(str(code), 0.0)

    # --- Filter advan_df to target_date ---
    mob = advan_df[pd.to_datetime(advan_df['date_range_start']) == target_date].copy()
    mob['cbsa_orig'] = mob['cbsa_orig'].astype(str)
    mob['cbsa_dest'] = mob['cbsa_dest'].astype(str)

    # --- Fixed CBSA universe: HHS region order, then west → east within each region ---
    cbsa_set = []
    _hhs_regions_used, _hhs_boundaries, _hhs_midpoints = [], [], []
    for _r in [str(i) for i in range(1, 11)]:
        _region_cbsas = sorted(get_cbsa_list(hhs_region=_r), key=_geo_sort_key)
        if not _region_cbsas:
            continue
        _start = len(cbsa_set)
        cbsa_set.extend(_region_cbsas)
        _end = len(cbsa_set)
        _hhs_regions_used.append(_r)
        _hhs_boundaries.append(_end)
        _hhs_midpoints.append((_start + _end) / 2)

    # --- Filter importance_masks to target_date ---
    if importance_masks is not None:
        imp = importance_masks[
            pd.to_datetime(importance_masks['date']) == target_date
        ].copy()
        imp['cbsa_orig'] = imp['cbsa_orig'].astype(str)
        imp['cbsa_dest'] = imp['cbsa_dest'].astype(str)
        if importance_threshold is not None:
            imp = imp[imp['importance'] >= importance_threshold]
    else:
        imp = None

    # --- Build mobility pivot: log(1 + total flow) ---
    mob_sub = mob[
        mob['cbsa_orig'].isin(cbsa_set) &
        mob['cbsa_dest'].isin(cbsa_set)
    ]
    mob_agg = (
        mob_sub.groupby(['cbsa_orig', 'cbsa_dest'])['visitor_home_aggregation']
        .sum()
        .reset_index()
    )
    mob_agg['log_flow'] = np.log1p(mob_agg['visitor_home_aggregation'])
    mob_matrix = (
        mob_agg.pivot(index='cbsa_orig', columns='cbsa_dest', values='log_flow')
        .reindex(index=cbsa_set, columns=cbsa_set)
        .fillna(0)
    )

    # --- Build importance pivot ---
    if imp is not None:
        imp_matrix = (
            imp.pivot(index='cbsa_orig', columns='cbsa_dest', values='importance')
            .reindex(index=cbsa_set, columns=cbsa_set)
            .fillna(0)
        )
    else:
        imp_matrix = None

    # --- Diagonal mask ---
    diag_mask = None
    if mask_diagonal:
        diag_mask = pd.DataFrame(
            np.eye(len(cbsa_set), dtype=bool),
            index=cbsa_set, columns=cbsa_set,
        )

    # --- Figure: GridSpec for equal-sized panels + dedicated colorbar column ---
    n_panels = 2 if imp_matrix is not None else 1
    if axes is None:
        fig = plt.figure(figsize=(8 * n_panels, 8), dpi=150, layout='constrained')
        gs = GridSpec(
            2, n_panels,
            height_ratios=[1, 0.02],
            figure=fig,
        )
        _plot_axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
        cbar_ax = fig.add_subplot(gs[1, :])
    else:
        fig = None
        _plot_axes, cbar_ax = list(axes[:-1]), axes[-1]

    # --- Shared PowerNorm colorscale ---
    _mob_vmax = mob_matrix.values.max()
    _norm = mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=_mob_vmax)

    # --- Helpers for region annotations ---
    def _draw_region_lines(ax):
        for _b in _hhs_boundaries[:-1]:
            ax.axhline(_b, color='black', linewidth=0.6, zorder=10)
            ax.axvline(_b, color='black', linewidth=0.6, zorder=10)

    def _add_region_labels(ax, y_labels=True):
        if y_labels:
            ax.set_yticks(_hhs_midpoints)
            ax.set_yticklabels(
                [f'HHS {r}' for r in _hhs_regions_used], fontsize=8,
            )
        else:
            ax.set_yticks([])
        ax.set_xticks(_hhs_midpoints)
        ax.set_xticklabels(
            [f'HHS {r}' for r in _hhs_regions_used], fontsize=8,
            rotation=45, ha='right',
        )

    # --- Left panel: full mobility matrix ---
    sns.heatmap(
        mob_matrix,
        ax=_plot_axes[0],
        mask=diag_mask,
        cmap='Blues',
        norm=_norm,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        linewidths=0.0,
        square=True,
    )
    _draw_region_lines(_plot_axes[0])
    _add_region_labels(_plot_axes[0], y_labels=show_y_labels)
    _full_nonzero = mob_matrix.values.astype(bool)
    if mask_diagonal:
        np.fill_diagonal(_full_nonzero, False)
    _n_full_edges = int(_full_nonzero.sum())
    _plot_axes[0].set_title('')
    _plot_axes[0].set_xlabel('Destination CBSA', fontsize=9)
    _plot_axes[0].set_ylabel('Origin CBSA' if show_y_labels else '', fontsize=9)

    # --- Right panel: mobility filtered to subgraph edges ---
    if imp_matrix is not None:
        _n_edges = int((imp_matrix >= 0.5).sum().sum())
        _imp_mask = (imp_matrix < 0.5) if diag_mask is None else ((imp_matrix < 0.5) | diag_mask)
        sns.heatmap(
            mob_matrix,
            ax=_plot_axes[1],
            mask=_imp_mask,
            cmap='Blues',
            norm=_norm,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            linewidths=0.0,
            square=True,
        )
        _draw_region_lines(_plot_axes[1])
        _add_region_labels(_plot_axes[1], y_labels=False)
        _plot_axes[1].set_title('')
        _plot_axes[1].set_xlabel('Destination CBSA', fontsize=9)
        _plot_axes[1].set_ylabel('')

    # --- Shared colorbar ---
    if draw_colorbar:
        _sm = mcm.ScalarMappable(cmap='Blues', norm=_norm)
        _sm.set_array([])
        cbar = cbar_ax.figure.colorbar(_sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('log(1 + visitor flow)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    if fig is not None:
        fig.suptitle(
            f'CBSA mobility matrix ({target_date.strftime("%Y-%m-%d")})',
            fontsize=10, y=1.02,
        )
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    return fig
