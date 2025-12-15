import pandas as pd
import numpy as np
import dash
from dash import html, dcc, Input, Output, State, ClientsideFunction, Patch
from flask_caching import Cache
from flask_compress import Compress
from plotly import graph_objects as go
import geopandas as gpd
import json
import plotly.express as px


from config import (
    STATES, DATASET_LABELS, DATASET_TITLES, REVERSED_DATASETS,
    DATASET_FORMAT_CONFIG, METRO_ONLY_DATASETS, MONO_COLORS, TS_COLORS
)

from helpers import (
    get_data_path, get_file_path, get_primary_state, human_format,
    get_shortened_cbsa_name, get_color_indices, format_metric_vectorized,
    get_color_vectorized
)

CBSA_DATA_DIR = get_data_path('cbsa_data')

AGG_DATA_PATH = get_file_path('agg_data')
AGG_DATA_METRO_PATH = get_file_path('agg_data_metro')
CBSA_METADATA_PATH = get_file_path('metadata')

GEOMETRIES_PATH = get_file_path('geometries')
STATE_BOUNDARIES_PATH = get_file_path('state_boundaries')
STATE_FIPS_CODES = get_file_path('state_fips_codes')

with open(STATE_BOUNDARIES_PATH) as geofile:
    geojson_data = json.load(geofile)


agg_metrics_all = pd.read_parquet(AGG_DATA_PATH)
agg_metrics_metro = pd.read_parquet(AGG_DATA_METRO_PATH)
geometries = gpd.read_parquet(GEOMETRIES_PATH)


def get_agg_metrics(metros_only: bool = False) -> pd.DataFrame:
    """Return the appropriate aggregate metrics based on filter."""
    return agg_metrics_metro if metros_only else agg_metrics_all

metric_dataframes: dict[str, pd.DataFrame] = {}
for fp in CBSA_DATA_DIR.glob('*.parquet'):
    if "metadata" in fp.parts:
        continue
    elif "aggregate" in fp.parts:
        continue
    key = fp.stem
    df = pd.read_parquet(fp)
    metric_dataframes[key] = df

cbsa_meta = pd.read_parquet(CBSA_METADATA_PATH)

for key, df in metric_dataframes.items():
    if 'cbsa_title' not in df.columns:
        df = df.merge(cbsa_meta[['cbsa_title']], left_on='cbsa_code', right_index=True, how='left')
        metric_dataframes[key] = df

state_fips_df = pd.read_csv(STATE_FIPS_CODES)

CBSA_GEOJSON = {
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'geometry': geom.__geo_interface__,
            'id': int(code)
        }
        for code, geom in zip(geometries['cbsa_code'], geometries['geometry'])
    ]
}

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "Real Estate Market Analysis Dashboard"
server = app.server
Compress(app.server)

cbsa_state_options = [
    {"label": str(STATES[state]), "value": str(state)}
    for state in STATES
]


app.layout = html.Div(
    [
        dcc.Store(id='map-data-store'),
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("dash-logo.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Real Estate Market Health",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "CBSA Analysis", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("by: TJ Norman", id="learn-more-button"),
                            href="https://tjnorman.xyz/",
                        )
                    ],
                    className="one-third column",
                    id="button",
                )
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Filter by date range:",
                            className="control_label",
                        ),
                        dcc.RangeSlider(
                            id="year_slider",
                            min=2016,
                            max=2025,
                            marks={i: "{}".format(i) for i in range(2016, 2026, 1)},
                            step=1,
                            value=[2023, 2025],
                            className="dcc_control",
                            updatemode='mouseup'
                        ),
                        html.P("Filter by State:", className="control_label"),
                        dcc.RadioItems(
                            id="cbsa_state_selector",
                            options=[
                                {"label": "All ", "value": "all"},
                                {"label": "CONUS ", "value": "active"},
                                {"label": "Customize ", "value": "custom"},
                            ],
                            value="all",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        dcc.Dropdown(
                            id="cbsa_states",
                            options=cbsa_state_options,
                            multi=True,
                            value=list(STATES.keys()),
                            className="dcc_control",
                        ),
                        dcc.Checklist(
                            id="metro_selector",
                            options=[{"label": "Metros Only", "value": "Metro"}],
                            className="dcc_control",
                            value=[],
                        ),
                        html.P("Select Real Estate Metric:", className="control_label"),
                        dcc.Dropdown(
                            id="dataset-dropdown",
                            options=[{'label': DATASET_TITLES[key], 'value': key}
                                   for key in DATASET_TITLES.keys()],
                            value='median_days_on_market',
                            className="dcc_control",
                        ),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="top10cbsas_graph", responsive=True)],
                            id="top10cbsasGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="main_graph", responsive=True,
                    style={"height": "70vh"})],
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="individual_graph", responsive=True,
                    style={"height": "70vh"})],
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="area_graph", responsive=True,
                    style={"height": "55vh"})],
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="price_decrease_ratio_graph", responsive=True,
                    style={"height": "55vh"})],
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        )

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("top10cbsas_graph", "figure")],
)

@app.callback(
    Output("cbsa_states", "style"),
    [Input("cbsa_state_selector", "value")]
)
def toggle_state_dropdown_visibility(selector):
    if selector == "custom":
        return {"display": "block"}
    return {"display": "none"}

@app.callback(
    Output("cbsa_states", "value"),
    [Input("cbsa_state_selector", "value")]
)
def reset_state_selection(selector):
    """Reset state dropdown when filter mode changes."""
    return []

cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 3600,
    'CACHE_THRESHOLD': 1000
})

@app.callback(
    Output("metro_selector", "value"),
    [Input("dataset-dropdown", "value")]
)
def update_metro_selector(selected_dataset):
    """Update the metro selector based on the selected dataset."""
    if selected_dataset in METRO_ONLY_DATASETS:
        return ["Metro"]
    return []


def _filter_cbsa_codes(state_selector, selected_states, metros_only):
    """Filter CBSA codes using index-native operations.
    
    Args:
        state_selector (str): Type of state filtering ('custom', 'active', 'all').
        selected_states (list[str]): List of state codes for custom filtering.
        metros_only (bool): If True, filter to Metro CBSAs only.
    
    Returns:
        list[int]: List of CBSA codes.
    """
    meta = cbsa_meta.copy()
    if meta.empty:
        return []
    
    if state_selector == "custom" and selected_states:
        masks = [meta['state'].str.contains(state, regex=False, na=False) 
                 for state in selected_states]
        mask = np.logical_or.reduce(masks) if masks else pd.Series([False] * len(meta))
        meta = meta[mask]
    elif state_selector == "active":
        meta = meta[~meta['state'].isin(['HI', 'AK'])]
    
    if metros_only and 'cbsa_type' in meta.columns:
        meta = meta[meta['cbsa_type'] == 'Metro']
    
    return meta.index.dropna().astype(int).unique().tolist()


def generate_map_frames(selected_dataset, df_slice, base_col, date_cols, year_range=None):
    """Generate map frames with optimized vectorized operations.
    
    Args:
        selected_dataset (str): The dataset name.
        df_slice (pd.DataFrame): The filtered dataframe slice.
        base_col (str): The base column name.
        date_cols (list[datetime]): List of date columns to generate frames for.
        year_range (tuple[int, int] | None): Optional tuple (start_year, end_year) to limit frame generation.
    
    Returns:
        list[go.Frame]: List of plotly Frame objects.
    """
    # Filter date_cols to only include dates within year_range
    if year_range:
        start_year, end_year = year_range
        date_cols = [d for d in date_cols if start_year <= d.year <= end_year]
    
    date_mapping = {date: date.strftime('%Y-%m-%d') for date in date_cols}
    format_cfg = DATASET_FORMAT_CONFIG[selected_dataset]
    frames = []
    mom_col = f"{base_col}_mom"
    yoy_col = f"{base_col}_yoy"
    
    df_slice_dates = pd.to_datetime(df_slice['date'])
    
    for date in date_cols:
        d = df_slice[df_slice_dates == date].copy()
        if d.empty:
            continue
        
        if 'cbsa_title' not in d.columns:
            d = d.merge(cbsa_meta[['cbsa_title']], left_on='cbsa_code', right_index=True, how='left')
        
        mom_vals = d[mom_col] if mom_col in d.columns else pd.Series(index=d.index, dtype=float)
        yoy_vals = d[yoy_col] if yoy_col in d.columns else pd.Series(index=d.index, dtype=float)
        
        mom_fmt = format_metric_vectorized(mom_vals, decimals=2, suffix='%')
        yoy_fmt = format_metric_vectorized(yoy_vals, decimals=2, suffix='%')
        mom_colors = get_color_vectorized(mom_vals, selected_dataset, REVERSED_DATASETS)
        yoy_colors = get_color_vectorized(yoy_vals, selected_dataset, REVERSED_DATASETS)
        
        frame = go.Frame(
            data=[go.Choroplethmapbox(
                locations=d['cbsa_code'],
                z=d[base_col],
                text=d['cbsa_title'],
                customdata=np.column_stack((
                    d['cbsa_code'],
                    mom_fmt,
                    yoy_fmt,
                    mom_colors,
                    yoy_colors
                )),
                hovertemplate=(
                    "<b>%{text}</b><br><br>"
                    f"{DATASET_LABELS[selected_dataset]}: "
                    f"{format_cfg['prefix']}%{{z:{format_cfg['format']}}}{format_cfg['suffix']}<br>"
                    "M/M Change: <span style='color:%{customdata[3]}'>%{customdata[1]}</span><br>"
                    "Y/Y Change: <span style='color:%{customdata[4]}'>%{customdata[2]}</span><br>"
                    "<extra></extra>"
                ),
            )],
            name=date_mapping[date]
        )
        frames.append(frame)
    return frames


@cache.memoize(timeout=3600)
def _generate_frames_cached(selected_dataset, cbsa_codes_tuple, year_range_tuple):
    """Cached frame generation to avoid recomputing identical requests.
    
    Args:
        selected_dataset (str): Dataset name.
        cbsa_codes_tuple (tuple[int]): Tuple of CBSA codes (must be hashable).
        year_range_tuple (tuple[int, int]): Tuple of (start_year, end_year).
    
    Returns:
        list[dict]: Serialized frames data.
    """
    df = metric_dataframes.get(selected_dataset, pd.DataFrame())
    if df.empty:
        return []
    
    start_year, end_year = year_range_tuple
    view = df[(pd.to_datetime(df['date']).dt.year >= start_year) & 
              (pd.to_datetime(df['date']).dt.year <= end_year)]
    
    if cbsa_codes_tuple:
        view = view[view['cbsa_code'].isin(list(cbsa_codes_tuple))]
    
    dates = sorted(pd.to_datetime(view['date']).unique())
    
    frames = generate_map_frames(selected_dataset, view, selected_dataset, dates, year_range_tuple)
    
    result = [{'data': frame.data, 'name': frame.name} for frame in frames]
    
    return result


@cache.memoize(timeout=3600)
def generate_map(selected_dataset, year_range=None, metros_only=False, selected_states=None, state_selector=None):
    """Generate choropleth map with optimization and caching.
    
    This function uses caching to avoid regenerating identical maps,
    and only generates frames for the visible year range.

    Args:
        selected_dataset (str): The dataset to visualize.
        year_range (list[int] | None): Range of years to include [start, end].
        metros_only (bool): If True, filter to Metro areas only.
        selected_states (list[str] | None): List of state codes for custom filtering.
        state_selector (str | None): Mode for state filtering ('all', 'active', 'custom').

    Returns:
        go.Figure: The constructed Plotly figure.
    """
    cbsa_codes_list = _filter_cbsa_codes(state_selector, selected_states, metros_only)
    cbsa_codes_tuple = tuple(sorted(cbsa_codes_list))
    year_range_tuple = tuple(year_range) if year_range else (2016, 2025)
    
    frame_data = _generate_frames_cached(selected_dataset, cbsa_codes_tuple, year_range_tuple)
    
    if not frame_data:
        return go.Figure()
    
    frames = [go.Frame(data=fd['data'], name=fd['name']) for fd in frame_data]
    
    # use latest data as initial display
    last_frame_trace = frames[-1].data[0]
    
    format_cfg = DATASET_FORMAT_CONFIG[selected_dataset]
    initial_trace = go.Choroplethmapbox(
        geojson=CBSA_GEOJSON,
        featureidkey='id',
        locations=last_frame_trace.locations,
        z=last_frame_trace.z,
        colorscale='RdBu' if selected_dataset in REVERSED_DATASETS else 'RdBu_r',
        text=last_frame_trace.text,
        customdata=last_frame_trace.customdata,
        hovertemplate=(
            "<b>%{text}</b><br><br>"
            f"{DATASET_LABELS[selected_dataset]}: "
            f"{format_cfg['prefix']}%{{z:{format_cfg['format']}}}{format_cfg['suffix']}<br>"
            "M/M Change: <span style='color:%{customdata[3]}'>%{customdata[1]}</span><br>"
            "Y/Y Change: <span style='color:%{customdata[4]}'>%{customdata[2]}</span><br>"
            "<extra></extra>"
        ),
    )
    
    fig = go.Figure(
        data=[initial_trace],
        frames=frames
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(
            center={"lat": 38.0902, "lon": -95.99},
            zoom=3,
            layers=[{
                'sourcetype': 'geojson',
                'source': geojson_data,
                'type': 'line',
                'color': 'rgba(0, 0, 0, .5)',
                'opacity': .75
            }]
        ),
        autosize=True,
        title={
            'text': f"{DATASET_TITLES[selected_dataset]} by Core-Based Statistical Area",
            'x': 0.5,
            'xanchor': 'center'
        },
        margin={"r":0,"t":45,"l":0,"b":0},
        hoverlabel=dict(bgcolor="rgba(0, 0, 0, 0.8)"),
        sliders=[{
            'active': len(frames) - 1,
            'currentvalue': {'prefix': 'Date: ', 'visible': True, 'xanchor': 'right'},
            'pad': {'b': 10, 't': 20},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [[frame.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': frame.name,
                    'method': 'animate'
                } for frame in frames
            ]
        }]
    )
    
    return fig

# Clientside callback for map updates
app.clientside_callback(
    """
    function(mapData, selectedDataset) {
        if (!mapData) {
            return {
                data: [],
                layout: {
                    title: 'No data available'
                }
            };
        }
        return mapData;
    }
    """,
    Output("main_graph", "figure"),
    [Input("map-data-store", "data")]
)

@app.callback(
    Output("map-data-store", "data"),
    [Input("dataset-dropdown", "value"),
     Input("year_slider", "value"),
     Input("metro_selector", "value"),
     Input("cbsa_states", "value"),
     Input("cbsa_state_selector", "value")],
    [State("map-data-store", "data")]
)
def update_map_data(selected_dataset, year_range, metro_filter, selected_states, state_selector, current_data):
    """Updates the map data store, using partial updates (Patch) where possible.

    Args:
        selected_dataset (str): Currently selected metric.
        year_range (list[int]): Selected year range [start, end].
        metro_filter (list[str]): Checkbox value for metro filtering.
        selected_states (list[str]): Selected states for custom filter.
        state_selector (str): State filter mode ('all', 'active', 'custom').
        current_data (dict): Current data in the store.

    Returns:
        dict | Patch: The new map data or a Patch object for incremental updates.
    """
    ctx = dash.callback_context
    metros_only = 'Metro' in metro_filter if metro_filter else False
    
    if not ctx.triggered or current_data is None:
        return generate_map(selected_dataset, year_range, metros_only, selected_states, state_selector).to_dict()
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id in ["dataset-dropdown", "year_slider"]:
        return generate_map(selected_dataset, year_range, metros_only, selected_states, state_selector).to_dict()
    
    # Try incremental updates for state/metro filter changes
    try:
        patch = Patch()
        new_map = generate_map(selected_dataset, year_range, metros_only, selected_states, state_selector).to_dict()
        
        patch["data"] = new_map.get("data", [])
        patch["frames"] = new_map.get("frames", [])
        
        return patch

    except Exception as e:
        print(f"Patch update failed: {e}")
        return generate_map(selected_dataset, year_range, metros_only, selected_states, state_selector).to_dict()

@app.callback(
    Output("individual_graph", "figure"),
    [Input("dataset-dropdown", "value"),
     Input("main_graph", "hoverData"),
     Input("year_slider", "value"),
     Input("metro_selector", "value")]
)
def update_time_series(selected_dataset, hover_data, year_range, metro_filter):
    """Updates the time series graph based on hover data.

    Args:
        selected_dataset (str): The currently selected metric.
        hover_data (dict): Data from the hover event on the main map.
        year_range (list[int]): Selected year range [start, end].
        metro_filter (list[str]): Metro filter selection.

    Returns:
        go.Figure: Time series figure for the hovered CBSA vs US/State medians.
    """
    if hover_data is None:
        return px.line(title="Hover over a CBSA to see time series data")
    
    metros_only = 'Metro' in metro_filter if metro_filter else False
    agg_metrics = get_agg_metrics(metros_only)

    cbsa_code = int(hover_data['points'][0]['customdata'][0])

    base = selected_dataset
    df = metric_dataframes.get(base, pd.DataFrame())
    if df.empty:
        return px.line(title="No data available for selected CBSA")
    view = df[df['cbsa_code'] == cbsa_code][['date', base] + ([f"{base}_mom"] if f"{base}_mom" in df.columns else []) + ([f"{base}_yoy"] if f"{base}_yoy" in df.columns else [])]
    if year_range:
        start_year, end_year = year_range
        view = view[(view['date'].dt.year >= start_year) & (view['date'].dt.year <= end_year)]
    if view.empty:
        return px.line(title="No data available for selected CBSA")

    dates = sorted(view['date'].unique())
    cbsa_base = view[['date', base]].rename(columns={base: 'value'})
    cbsa_mom = view[['date', f"{base}_mom"]].rename(columns={f"{base}_mom": 'value'}) if f"{base}_mom" in view.columns else pd.DataFrame({'date': dates, 'value': np.nan})
    cbsa_yoy = view[['date', f"{base}_yoy"]].rename(columns={f"{base}_yoy": 'value'}) if f"{base}_yoy" in view.columns else pd.DataFrame({'date': dates, 'value': np.nan})

    us_base = agg_metrics[(agg_metrics['metric'] == selected_dataset) & (agg_metrics['fipsstatecode'] == 0)][['date', 'value']]
    us_mom = agg_metrics[(agg_metrics['metric'] == f"{selected_dataset}_mom") & (agg_metrics['fipsstatecode'] == 0)][['date', 'value']]
    us_yoy = agg_metrics[(agg_metrics['metric'] == f"{selected_dataset}_yoy") & (agg_metrics['fipsstatecode'] == 0)][['date', 'value']]
    if year_range:
        for df in (us_base, us_mom, us_yoy):
            df.loc[:, 'date'] = pd.to_datetime(df['date'])
            df.dropna(subset=['date'], inplace=True)
            df.query("@start_year <= date.dt.year <= @end_year", inplace=True)

    meta_row = cbsa_meta.loc[cbsa_code] if cbsa_code in cbsa_meta.index else None
    if meta_row is not None:
        cbsa_name = meta_row['cbsa_title']
        cbsa_type = meta_row['cbsa_type'] if 'cbsa_type' in cbsa_meta.columns else ''
        cbsa_primary_state = meta_row['primary_state'] if 'primary_state' in cbsa_meta.columns else get_primary_state(meta_row.get('state', ''))
    else:
        cbsa_name = str(cbsa_code)
        cbsa_type = ''
        cbsa_primary_state = ''
    
    state_row = state_fips_df[state_fips_df['state'] == cbsa_primary_state]
    state_base = pd.DataFrame(columns=['date', 'value'])
    state_mom = pd.DataFrame(columns=['date', 'value'])
    state_yoy = pd.DataFrame(columns=['date', 'value'])
    if not state_row.empty:
        primary_state_code = int(state_row.iloc[0]['fipsstatecode'])
        state_base = agg_metrics[(agg_metrics['metric'] == selected_dataset) & (agg_metrics['fipsstatecode'] == primary_state_code)][['date', 'value']]
        state_mom = agg_metrics[(agg_metrics['metric'] == f"{selected_dataset}_mom") & (agg_metrics['fipsstatecode'] == primary_state_code)][['date', 'value']]
        state_yoy = agg_metrics[(agg_metrics['metric'] == f"{selected_dataset}_yoy") & (agg_metrics['fipsstatecode'] == primary_state_code)][['date', 'value']]
        if year_range:
            for df in (state_base, state_mom, state_yoy):
                df.loc[:, 'date'] = pd.to_datetime(df['date'])
                df.dropna(subset=['date'], inplace=True)
                df.query("@start_year <= date.dt.year <= @end_year", inplace=True)

    fig = go.Figure()

    # US median trace
    us_label = 'US Metro Median' if metros_only else 'US Median'
    if not us_base.empty:
        us_join = pd.DataFrame({'date': dates}).merge(us_base, on='date', how='left')
        us_join = us_join.merge(us_mom.rename(columns={'value': 'mom'}), on='date', how='left')
        us_join = us_join.merge(us_yoy.rename(columns={'value': 'yoy'}), on='date', how='left')
        fig.add_trace(go.Scatter(
                x=us_join['date'],
                y=us_join['value'],
            name=us_label,
            mode='lines',
            line=dict(color='black', width=2),
                customdata=np.column_stack([
                    format_metric_vectorized(us_join['mom'], decimals=2, suffix='%'),
                    format_metric_vectorized(us_join['yoy'], decimals=2, suffix='%'),
                    get_color_vectorized(us_join['mom'], selected_dataset, REVERSED_DATASETS),
                    get_color_vectorized(us_join['yoy'], selected_dataset, REVERSED_DATASETS),
                ]),
            hovertemplate=(
                f"<b>{us_label}</b><br>"
                f"{DATASET_LABELS[selected_dataset]}: %{{y}}<br>"
                "Changes: <span style='color:%{customdata[2]}'>M/M: %{customdata[0]}</span> | "
                "<span style='color:%{customdata[3]}'>Y/Y: %{customdata[1]}</span><extra></extra>"
            )
        ))

    # State median trace
    state_label = f'{cbsa_primary_state} Metro Median' if metros_only else f'{cbsa_primary_state} Median'
    if not state_base.empty:
        st_join = pd.DataFrame({'date': dates}).merge(state_base, on='date', how='left')
        st_join = st_join.merge(state_mom.rename(columns={'value': 'mom'}), on='date', how='left')
        st_join = st_join.merge(state_yoy.rename(columns={'value': 'yoy'}), on='date', how='left')
        fig.add_trace(go.Scatter(
            x=st_join['date'],
            y=st_join['value'],
            name=state_label,
            mode='lines',
            line=dict(color='grey', width=2, dash='dash'),
            customdata=np.column_stack([
                format_metric_vectorized(st_join['mom'], decimals=2, suffix='%'),
                format_metric_vectorized(st_join['yoy'], decimals=2, suffix='%'),
                get_color_vectorized(st_join['mom'], selected_dataset, REVERSED_DATASETS),
                get_color_vectorized(st_join['yoy'], selected_dataset, REVERSED_DATASETS),
            ]),
            hovertemplate=(
                f"<b>{state_label}</b><br>"
                f"{DATASET_LABELS[selected_dataset]}: %{{y}}<br>"
                "Changes: <span style='color:%{customdata[2]}'>M/M: %{customdata[0]}</span> | "
                "<span style='color:%{customdata[3]}'>Y/Y: %{customdata[1]}</span><extra></extra>"
            )
        ))

    # CBSA trace
    mom_join = pd.DataFrame({'date': dates}).merge(cbsa_mom.rename(columns={'value': 'mom'}), on='date', how='left')
    yoy_join = pd.DataFrame({'date': dates}).merge(cbsa_yoy.rename(columns={'value': 'yoy'}), on='date', how='left')
    cbsa_join = pd.DataFrame({'date': dates}).merge(cbsa_base, on='date', how='left')
    cbsa_join = cbsa_join.merge(mom_join, on='date', how='left').merge(yoy_join, on='date', how='left')

    legend_name = get_shortened_cbsa_name(cbsa_name, cbsa_type)
    fig.add_trace(go.Scatter(
        x=cbsa_join['date'],
        y=cbsa_join['value'],
        name=legend_name,
        mode='lines',
        line=dict(color=TS_COLORS[0], width=2),
        customdata=np.column_stack([
            format_metric_vectorized(cbsa_join['mom'], decimals=2, suffix='%'),
            format_metric_vectorized(cbsa_join['yoy'], decimals=2, suffix='%'),
            get_color_vectorized(cbsa_join['mom'], selected_dataset, REVERSED_DATASETS),
            get_color_vectorized(cbsa_join['yoy'], selected_dataset, REVERSED_DATASETS),
        ]),
        hovertemplate=(
            f"<b>{cbsa_name}</b><br>"
            f"{DATASET_LABELS[selected_dataset]}: %{{y}}<br>"
            "Changes: <span style='color:%{customdata[2]}'>M/M: %{customdata[0]}</span> | "
            "<span style='color:%{customdata[3]}'>Y/Y: %{customdata[1]}</span><extra></extra>"
        )
    ))

    fig.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=1, xanchor='right', x=1,
            yref='paper', xref='paper'),
        legend_title_text='',
        xaxis_title='Date',
        yaxis_title=DATASET_LABELS[selected_dataset],
        hovermode='x',
        margin=dict(t=100, b=40, l=60, r=40),
        hoverlabel=dict(
            bgcolor="rgba(0, 0, 0, 0.8)",
            font_color="white"
        ),
        title={
            'text': f"<b>{DATASET_LABELS[selected_dataset]} Comparison: </b><br><span style='font-size:0.95em'>{cbsa_name}</span>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    return fig

@app.callback(
    Output("top10cbsas_graph", "figure"),
    [Input("dataset-dropdown", "value"),
     Input("year_slider", "value"),
     Input("metro_selector", "value"),
     Input("cbsa_states", "value"),
     Input("cbsa_state_selector", "value")]
)
def update_top_cbsas(selected_dataset, year_range, metro_filter, selected_states, state_selector):
    """Updates the 'Top CBSAs' chart based on current filters.

    Args:
        selected_dataset (str): Selected metric.
        year_range (list[int]): Selected year range.
        metro_filter (list[str]): Metro filter selection.
        selected_states (list[str]): Selected states.
        state_selector (str): State filter mode.

    Returns:
        go.Figure: Chart showing top CBSAs by population.
    """
    if not year_range:
        return go.Figure()

    start_year, end_year = year_range
    metros_only = 'Metro' in metro_filter if metro_filter else False
    agg_metrics = get_agg_metrics(metros_only)

    filtered_cbsa_codes = _filter_cbsa_codes(state_selector, selected_states, metros_only)
    filtered_cbsas = cbsa_meta.loc[filtered_cbsa_codes] if filtered_cbsa_codes else cbsa_meta.iloc[:0]
    
    date_range_years = end_year - start_year
    show_markers = date_range_years < 5
    n_cbsas = 3 if date_range_years >= 5 else min(5, len(filtered_cbsas))
    if n_cbsas < 5 and n_cbsas > 3:
        n_cbsas = 3
    
    # Calculate evenly spaced color indices for maximum contrast
    color_indices = get_color_indices(n_cbsas)
    
    top_n_cbsas = filtered_cbsas.nlargest(n_cbsas, 'population_16yo_plus')
    
    metric_df = metric_dataframes.get(selected_dataset, pd.DataFrame())
    date_mask = (metric_df['date'].dt.year >= start_year) & (metric_df['date'].dt.year <= end_year)
    date_columns = sorted(metric_df.loc[date_mask, 'date'].unique())

    us_base = agg_metrics[(agg_metrics['metric'] == selected_dataset) & (agg_metrics['fipsstatecode'] == 0)]
    us_base = us_base[us_base['date'].isin(date_columns)].sort_values('date')
    us_mom = agg_metrics[(agg_metrics['metric'] == f"{selected_dataset}_mom") & (agg_metrics['fipsstatecode'] == 0)]
    us_mom = us_mom[us_mom['date'].isin(date_columns)].sort_values('date')
    us_yoy = agg_metrics[(agg_metrics['metric'] == f"{selected_dataset}_yoy") & (agg_metrics['fipsstatecode'] == 0)]
    us_yoy = us_yoy[us_yoy['date'].isin(date_columns)].sort_values('date')

    fig = go.Figure()
    us_label = 'US Metro Median' if metros_only else 'US Median'

    # US Median trace
    if not us_base.empty:
        us_join = us_base[['date', 'value']].merge(us_mom[['date', 'value']].rename(columns={'value': 'mom'}), on='date', how='left')
        us_join = us_join.merge(us_yoy[['date', 'value']].rename(columns={'value': 'yoy'}), on='date', how='left')
        fig.add_trace(go.Scatter(
                x=us_join['date'],
                y=us_join['value'],
            name=us_label,
            mode='lines+markers' if show_markers else 'lines',
            line=dict(color='black', width=3, dash='solid'),
            marker=dict(size=4) if show_markers else None,
                customdata=np.column_stack([
                    format_metric_vectorized(us_join['mom'], decimals=1, suffix='%'),
                    format_metric_vectorized(us_join['yoy'], decimals=1, suffix='%'),
                    get_color_vectorized(us_join['mom'], selected_dataset, REVERSED_DATASETS),
                    get_color_vectorized(us_join['yoy'], selected_dataset, REVERSED_DATASETS),
                ]),
            hovertemplate=(
                f"<b>{us_label}</b><br>"
                    f"{DATASET_FORMAT_CONFIG[selected_dataset]['prefix']}%{{y:{DATASET_FORMAT_CONFIG[selected_dataset]['format']}}}{DATASET_FORMAT_CONFIG[selected_dataset]['suffix']} "
                "<b>[ </b>M/M: <span style='color:%{customdata[2]}'>%{customdata[0]} </span><b>|</b> "
                "Y/Y: <span style='color:%{customdata[3]}'>%{customdata[1]}</span><b> ]</b>"
                "<extra></extra>"
            )
        ))

    # CBSA traces
    for idx, cbsa_row in enumerate(top_n_cbsas.itertuples()):
        cbsa_code = int(cbsa_row.Index)
        cbsa_title = cbsa_row.cbsa_title
        meta_row = cbsa_meta.loc[cbsa_code] if cbsa_code in cbsa_meta.index else None
        cbsa_type = meta_row['cbsa_type'] if meta_row is not None and 'cbsa_type' in cbsa_meta.columns else ''

        metric_df = metric_dataframes.get(selected_dataset, pd.DataFrame())
        
        cbsa_data = metric_df[
            (metric_df['cbsa_code'] == cbsa_code) & 
            (metric_df['date'].isin(date_columns))
        ].sort_values('date')
        
        if cbsa_data.empty:
            continue
            
        # Extract base, mom, and yoy values with explicit column names
        cbsa_base_values = cbsa_data[['date', selected_dataset]] if selected_dataset in cbsa_data.columns else pd.DataFrame()
        cbsa_mom_values = cbsa_data[['date', f"{selected_dataset}_mom"]] if f"{selected_dataset}_mom" in cbsa_data.columns else None
        cbsa_yoy_values = cbsa_data[['date', f"{selected_dataset}_yoy"]] if f"{selected_dataset}_yoy" in cbsa_data.columns else None
        
        if cbsa_base_values.empty:
            continue
            
        # Create plotting data structure with standardized column names
        cbsa_join = pd.DataFrame({
            'date': cbsa_base_values['date'],
            'value': cbsa_base_values[selected_dataset],
            'mom': cbsa_mom_values[f"{selected_dataset}_mom"] if cbsa_mom_values is not None else np.nan,
            'yoy': cbsa_yoy_values[f"{selected_dataset}_yoy"] if cbsa_yoy_values is not None else np.nan
        })

        legend_cbsa_type = cbsa_type if state_selector == "custom" else ""
        legend_name = get_shortened_cbsa_name(cbsa_title, legend_cbsa_type)
        fig.add_trace(go.Scatter(
            x=cbsa_join['date'],
            y=cbsa_join['value'],
            name=legend_name,
            mode='lines+markers' if show_markers else 'lines',
            line=dict(color=MONO_COLORS[color_indices[idx]]),
            marker=dict(size=4) if show_markers else None,
            customdata=np.column_stack([
                format_metric_vectorized(cbsa_join['mom'], decimals=1, suffix='%'),
                format_metric_vectorized(cbsa_join['yoy'], decimals=1, suffix='%'),
                get_color_vectorized(cbsa_join['mom'], selected_dataset, REVERSED_DATASETS),
                get_color_vectorized(cbsa_join['yoy'], selected_dataset, REVERSED_DATASETS),
            ]),
            hovertemplate=(
                f"<b>{cbsa_title}</b><br>"
                f"{DATASET_FORMAT_CONFIG[selected_dataset]['prefix']}%{{y:{DATASET_FORMAT_CONFIG[selected_dataset]['format']}}}{DATASET_FORMAT_CONFIG[selected_dataset]['suffix']} "
                "<b>[ </b>M/M: <span style='color:%{customdata[2]}'>%{customdata[0]} </span><b>|</b> "
                "Y/Y: <span style='color:%{customdata[3]}'>%{customdata[1]}</span><b> ]</b>"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title={
            'text': f'Top {n_cbsas} Most Populous CBSAs: {DATASET_LABELS[selected_dataset]}',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title=DATASET_LABELS[selected_dataset],
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.35,
            xanchor='center',
            x=0.5,
            itemwidth=30,
            borderwidth=1,
            bordercolor='black',
            bgcolor='rgba(255,255,255,0.9)'
        ),
        margin=dict(t=50, b=100, l=60, r=40),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="rgba(44, 44, 44, 1)",
            font_color="white",
            font=dict(size=11)
        )
    )
    
    return fig


@app.callback(
    Output("area_graph", "figure"),
    [Input("year_slider", "value"),
     Input("metro_selector", "value"),
     Input("cbsa_states", "value"),
     Input("cbsa_state_selector", "value")]
)
def update_area_chart(year_range, metro_filter, selected_states, state_selector):
    """Updates the 'Active Listings Composition' stacked area chart.
    
    Args:
        year_range (list[int]): Selected year range [start, end].
        metro_filter (list[str]): Metro filter selection.
        selected_states (list[str]): Selected states for custom filter.
        state_selector (str): State filter mode ('all', 'active', 'custom').

    Returns:
        go.Figure: Stacked area chart showing active listings composition.
    """
    if not year_range:
        return go.Figure()

    start_year, end_year = year_range
    metros_only = 'Metro' in metro_filter if metro_filter else False
    
    apply_filters = (state_selector == "custom" and selected_states) or state_selector == "active" or metros_only
    cbsa_list = _filter_cbsa_codes(state_selector, selected_states, metros_only) if apply_filters else None
    
    act_base = 'active_listing_count'
    dec_base = 'price_decrease_count'
    inc_base = 'price_increase_count'
    act_df = metric_dataframes.get(act_base, pd.DataFrame())
    dec_df = metric_dataframes.get(dec_base, pd.DataFrame())
    inc_df = metric_dataframes.get(inc_base, pd.DataFrame())
    
    mask_dates_act = (act_df['date'].dt.year >= start_year) & (act_df['date'].dt.year <= end_year)
    mask_dates_dec = (dec_df['date'].dt.year >= start_year) & (dec_df['date'].dt.year <= end_year)
    mask_dates_inc = (inc_df['date'].dt.year >= start_year) & (inc_df['date'].dt.year <= end_year)
    
    if cbsa_list is not None:
        mask_dates_act &= act_df['cbsa_code'].isin(cbsa_list)
        mask_dates_dec &= dec_df['cbsa_code'].isin(cbsa_list)
        mask_dates_inc &= inc_df['cbsa_code'].isin(cbsa_list)
    
    total_active = act_df[mask_dates_act].groupby('date')[act_base].sum()
    price_dec = dec_df[mask_dates_dec].groupby('date')[dec_base].sum()
    price_inc = inc_df[mask_dates_inc].groupby('date')[inc_base].sum()
    
    all_dates = sorted(total_active.index.union(price_dec.index).union(price_inc.index))
    data = {
        'Date': all_dates,
        'Total Active': [total_active.get(d, 0.0) for d in all_dates],
        'Price Decreases': [price_dec.get(d, 0.0) for d in all_dates],
        'Price Increases': [price_inc.get(d, 0.0) for d in all_dates],
    }
    data['Unchanged'] = [max(0.0, ta - pdv - piv) for ta, pdv, piv in zip(data['Total Active'], data['Price Decreases'], data['Price Increases'])]
    
    # round to the nearest 100 (to get clean 'K' suffix)
    def round_to_nearest_100(value):
        if value >= 10000:
            return round(value / 100) * 100 

        return value

    fig = go.Figure()
    
    hovertemplate = (
        '<b>%{fullData.name}:</b> %{customdata[0]}% (%{customdata[1]})'
        '<extra></extra>'
    )
    
    unchanged_hover = [[round((unch / total * 100), 1) if total > 0 else 0, human_format(unch)] 
                      for unch, total in zip(data['Unchanged'], data['Total Active'])]
    increases_hover = [[round((inc / total * 100), 1) if total > 0 else 0, human_format(inc)] 
                      for inc, total in zip(data['Price Increases'], data['Total Active'])]
    decreases_hover = [[round((dec / total * 100), 1) if total > 0 else 0, human_format(dec)] 
                      for dec, total in zip(data['Price Decreases'], data['Total Active'])]
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=[round_to_nearest_100(y) for y in data['Unchanged']],
        customdata=unchanged_hover,
        name='Unchanged',
        mode='none',
        stackgroup='one',
        fillcolor='rgb(149, 165, 166)',
        line=dict(width=0),
        hovertemplate=hovertemplate
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=[round_to_nearest_100(y) for y in data['Price Increases']],
        customdata=increases_hover,
        name='Price Increases',
        mode='none',
        stackgroup='one',
        fillcolor='rgb(46, 204, 113)',
        line=dict(width=0),
        hovertemplate=hovertemplate
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=[round_to_nearest_100(y) for y in data['Price Decreases']],
        customdata=decreases_hover,
        name='Price Decreases',
        mode='none',
        stackgroup='one',
        fillcolor='rgb(231, 76, 60)',
        line=dict(width=0),
        hovertemplate=hovertemplate
    ))
    
    fig.update_layout(
        title={
            'text': 'Active Listings Composition Over Time',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Date',
            title_standoff=25,
            showline=True,
            linewidth=1,
            linecolor='black',
            ticklabelstandoff=10
        ),
        yaxis=dict(
            title='Number of Listings',
            showline=True,
            linewidth=1,
            linecolor='black',
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0, # Anchor within chart
            xanchor='center',
            x=0.5,
            borderwidth=1, 
            bordercolor='black',
            bgcolor='rgba(255,255,255,0.9)' 
        ),
        margin=dict(t=50, b=100, l=60, r=40),
        hoverlabel=dict(namelength=-1),
    )
    
    return fig


@app.callback(
    Output("price_decrease_ratio_graph", "figure"),
    [Input("year_slider", "value"),
     Input("metro_selector", "value"),
     Input("cbsa_states", "value"),
     Input("cbsa_state_selector", "value")]
)
def update_price_decrease_ratio(year_range, metro_filter, selected_states, state_selector):
    """Updates the 'Price Decrease Ratio' line chart.
    
    Args:
        year_range (list[int]): Selected year range [start, end].
        metro_filter (list[str]): Metro filter selection.
        selected_states (list[str]): Selected states for custom filter.
        state_selector (str): State filter mode ('all', 'active', 'custom').

    Returns:
        go.Figure: Line chart showing price decrease percentage over time.
    """
    if not year_range:
        return go.Figure()

    start_year, end_year = year_range
    metros_only = 'Metro' in metro_filter if metro_filter else False

    filtered_cbsa_codes = _filter_cbsa_codes(state_selector, selected_states, metros_only)
    filtered_cbsas = cbsa_meta.loc[filtered_cbsa_codes] if filtered_cbsa_codes else cbsa_meta.iloc[:0]

    date_range_years = end_year - start_year
    show_markers = date_range_years < 3
    if date_range_years >= 3:
        n_cbsas = min(3, len(filtered_cbsas))
    else:
        n_cbsas = min(5, len(filtered_cbsas))
        if 3 < n_cbsas < 5:
            n_cbsas = 3

    color_indices = get_color_indices(n_cbsas)
    top_n_cbsas = filtered_cbsas.nlargest(n_cbsas, 'population_16yo_plus')

    ratio_base = 'price_decrease_ratio'
    df_ratio = metric_dataframes.get(ratio_base, pd.DataFrame())
    if df_ratio.empty:
        return go.Figure()
    df_ratio = df_ratio[(df_ratio['date'].dt.year >= start_year) & (df_ratio['date'].dt.year <= end_year)]

    date_columns = sorted(df_ratio['date'].unique())

    fig = go.Figure()

    us_series = df_ratio[df_ratio['cbsa_code'] == 0][['date', ratio_base]].rename(columns={ratio_base: 'value'}).sort_values('date')
    us_mom = df_ratio[df_ratio['cbsa_code'] == 0][['date', f"{ratio_base}_mom"]].rename(columns={f"{ratio_base}_mom": 'mom'}) if f"{ratio_base}_mom" in df_ratio.columns else pd.DataFrame({'date': date_columns, 'mom': np.nan})
    us_yoy = df_ratio[df_ratio['cbsa_code'] == 0][['date', f"{ratio_base}_yoy"]].rename(columns={f"{ratio_base}_yoy": 'yoy'}) if f"{ratio_base}_yoy" in df_ratio.columns else pd.DataFrame({'date': date_columns, 'yoy': np.nan})
    if not us_series.empty:
        us_join = pd.DataFrame({'date': date_columns}).merge(us_series, on='date', how='left').merge(us_mom, on='date', how='left').merge(us_yoy, on='date', how='left')
        fig.add_trace(go.Scatter(
            x=us_join['date'],
            y=us_join['value'],
            name='US Median',
            mode='lines+markers' if show_markers else 'lines',
            line=dict(color='black', width=3, dash='solid'),
            marker=dict(size=6) if show_markers else None,
            customdata=np.column_stack([
                format_metric_vectorized(us_join['mom'], decimals=1, suffix=' pp'),
                format_metric_vectorized(us_join['yoy'], decimals=1, suffix=' pp'),
                get_color_vectorized(us_join['mom'], ratio_base, REVERSED_DATASETS),
                get_color_vectorized(us_join['yoy'], ratio_base, REVERSED_DATASETS),
            ]),
            hovertemplate=(
                "<b>US Median</b><br>"
                "Price Decrease Ratio: %{y:.1f}% <b>[ </b>M/M: <span style='color:%{customdata[2]}'>%{customdata[0]} </span><b>|</b> "
                "Y/Y: <span style='color:%{customdata[3]}'>%{customdata[1]}</span><b> ]</b>"
                "<extra></extra>"
            )
        ))

    # Top CBSA lines
    for idx, cbsa_row in enumerate(top_n_cbsas.itertuples()):
        cbsa_code = int(cbsa_row.Index)
        cbsa_title = cbsa_row.cbsa_title
        cbsa_title_legend = cbsa_title.split('-')[0] if '-' in cbsa_title else cbsa_title.split(',')[0]

        base = df_ratio[(df_ratio['cbsa_code'] == cbsa_code)][['date', ratio_base]].rename(columns={ratio_base: 'value'}).sort_values('date')
        if base.empty:
            continue
        mom = df_ratio[(df_ratio['cbsa_code'] == cbsa_code)][['date', f"{ratio_base}_mom"]].rename(columns={f"{ratio_base}_mom": 'mom'}) if f"{ratio_base}_mom" in df_ratio.columns else pd.DataFrame({'date': date_columns, 'mom': np.nan})
        mom = mom[mom['date'].isin(date_columns)]
        yoy = df_ratio[(df_ratio['cbsa_code'] == cbsa_code)][['date', f"{ratio_base}_yoy"]].rename(columns={f"{ratio_base}_yoy": 'yoy'}) if f"{ratio_base}_yoy" in df_ratio.columns else pd.DataFrame({'date': date_columns, 'yoy': np.nan})
        yoy = yoy[yoy['date'].isin(date_columns)]

        join_df = pd.DataFrame({'date': date_columns}).merge(base, on='date', how='left').merge(mom, on='date', how='left').merge(yoy, on='date', how='left')
        fig.add_trace(go.Scatter(
            x=join_df['date'],
            y=join_df['value'],
            name=cbsa_title_legend,
            mode='lines+markers' if show_markers else 'lines',
            line=dict(color=MONO_COLORS[color_indices[idx]]),
            marker=dict(size=4) if show_markers else None,
            customdata=np.column_stack([
                format_metric_vectorized(join_df['mom'], decimals=1, suffix=' pp'),
                format_metric_vectorized(join_df['yoy'], decimals=1, suffix=' pp'),
                get_color_vectorized(join_df['mom'], ratio_base, REVERSED_DATASETS),
                get_color_vectorized(join_df['yoy'], ratio_base, REVERSED_DATASETS),
            ]),
            hovertemplate=(
                f"<b>{cbsa_title}</b><br>"
                "Price Decrease Ratio: %{y:.1f}% <b>[ </b>M/M: <span style='color:%{customdata[2]}'>%{customdata[0]}</span> <b>|</b> "
                "Y/Y: <span style='color:%{customdata[3]}'>%{customdata[1]}</span><b> ]</b>"
                "<extra></extra>"
            )
        ))

    title_text = f"<b>% of Active Listings with Price Decreases</b><br><span style='font-size:0.95em'> Top {n_cbsas} CBSAs</span>"
    if metros_only:
        title_text = f"<b>% of Active Listings with Price Decreases</b><br><span style='font-size:0.95em'> Top {n_cbsas} Metro CBSAs</span>"
    
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'y':0.95,
            'yanchor': 'top'

        },
        xaxis_title='Date',
        yaxis_title='Percentage of Listings',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.35,
            xanchor='center',
            x=0.5,
            font=dict(size=9),
            itemwidth=30,
            borderwidth=1,
            bordercolor='black',
            bgcolor='rgba(255,255,255,0.9)'
        ),
        margin=dict(t=50, b=75, l=60, r=40),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="rgba(44, 44, 44, 1)",
            font_color="white",
            font=dict(size=11)
        )
    )
    
    return fig


if __name__ == '__main__':
    print("\n Starting dashboard server...")
    app.run_server(port=8097)