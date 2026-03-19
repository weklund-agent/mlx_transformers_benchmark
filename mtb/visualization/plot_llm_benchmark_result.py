from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from natsort import natsorted

from mtb.visualization.symbol_and_color import (
    add_category_to_colormap,
    get_symbol_and_color_map,
)


def show_llm_benchmark_data(
    title: str,
    measurements: pd.DataFrame,
    dtypes: List[str] = ("bfloat16",),
    batch_sizes: List[int] = (1,),
    do_average_measurements: bool = True,
) -> go.Figure:
    """Visualize benchmark data in a single page.

    Args:
        title: Title of the benchmark task.
        measurements: DataFrame containing benchmark measurements.
        dtypes: Tuple of data types to show. One dtype = one row.
        batch_sizes: Tuple of batchsizes to show. One batchsize = one column.
        do_average_measurements: If False, show all individual measurements.

    Returns:
        The created figure.

    """
    frameworks = natsorted(measurements["framework_backend"].unique())
    for framework in frameworks:
        add_category_to_colormap(framework)

    std_columns = [
        "prompt_tps_std",
        "generation_tps_std",
        "prompt_time_sec_std",
        "generation_time_sec_std",
        "peak_memory_gib_std",
    ]
    metrics_of_interest = [
        "prompt_tps",
        "generation_tps",
        "prompt_time_sec",
        "generation_time_sec",
        "total_time_sec",
        "peak_memory_gib",
    ]
    y_metrics = {
        "prompt_time_sec": "Time to first token (s)",
        "generation_tps": "Generation speed (tokens/s)",
        "peak_memory_gib": "Peak memory (GiB)",
    }

    fig = sp.make_subplots(
        rows=len(dtypes),
        cols=len(batch_sizes) * len(y_metrics),
        subplot_titles=[
            f"{title}, {dtype}, B={batch_size}"
            for dtype in dtypes
            for title in y_metrics.values()
            for batch_size in batch_sizes
        ],
        horizontal_spacing=0.075,
        vertical_spacing=0.075,
    )

    color_map, symbol_map = get_symbol_and_color_map()

    for row, dtype in enumerate(dtypes):
        for col, batch_size in enumerate(batch_sizes):
            # Select data
            filtered_data = measurements[
                (measurements["dtype"] == dtype)
                & (measurements["batch_size"] == batch_size)
            ]

            # Add std columns with default 0.0 for older data without them
            for col_name in std_columns:
                if col_name not in filtered_data.columns:
                    filtered_data = filtered_data.copy()
                    filtered_data[col_name] = 0.0

            if do_average_measurements:
                filtered_data = filtered_data[
                    [
                        "framework_backend",
                        "batch_size",
                        "num_prompt_tokens",
                    ]
                    + metrics_of_interest
                    + std_columns
                ]
                filtered_data = (
                    filtered_data.groupby(
                        ["framework_backend", "batch_size", "num_prompt_tokens"],
                        observed=True,
                    )
                    .mean()
                    .reset_index()
                )

            # Show
            if not filtered_data.empty:
                filtered_data = filtered_data.sort_values(
                    by=["framework_backend", "num_prompt_tokens", "batch_size"],
                )
                for col_offset, y_metric_name in enumerate(y_metrics):
                    row_index = row + 1
                    column_index = col * len(y_metrics) + col_offset + 1

                    scatter = px.scatter(
                        filtered_data,
                        x="num_prompt_tokens",
                        y=y_metric_name,
                        color="framework_backend",
                        symbol="framework_backend",
                        category_orders={"framework_backend": frameworks},
                        color_discrete_map=color_map,
                        symbol_map=symbol_map,
                        custom_data=["batch_size"] + metrics_of_interest + std_columns,
                        title=f"dtype: {dtype}, batch_size: {batch_size}",
                    )
                    for trace in scatter["data"]:
                        fig.add_trace(trace, row=row_index, col=column_index)
                        fig.update_yaxes(
                            row=row_index,
                            col=column_index,
                            title_text=y_metrics[y_metric_name],
                        )
            else:
                # Disable the subplot, show text "no data available"
                for col_offset in range(len(y_metrics)):
                    row_index = row + 1
                    column_index = col * len(y_metrics) + col_offset + 1

                    # disable the subplot
                    fig.update_xaxes(
                        row=row_index,
                        col=column_index,
                        visible=False,
                    )
                    fig.update_yaxes(
                        row=row_index,
                        col=column_index,
                        visible=False,
                    )
                    fig.add_annotation(
                        text="No data available",
                        row=row_index,
                        col=column_index,
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14),
                    )

    # Update x and y axes layouts for all subplots
    fig.update_xaxes(
        type="log",
        tickvals=[4096, 2048, 1024, 512, 256, 128, 64, 32],
        ticktext=["4096", "2048", "1024", "512", "256", "128", "64", "32"],
    )
    fig.update_xaxes(
        row=len(dtypes),
        title_text="Num prompt tokens",
    )
    fig.update_yaxes(
        type="log",
        tickformat=".2g",
    )

    # Optimize legend entries, layout
    legend_entries = set()
    for trace in fig.data:
        if trace.name not in legend_entries:
            legend_entries.add(trace.name)
        else:
            trace.showlegend = False

    fig.update_layout(
        height=800,
        width=1600,
        title_text=f"Benchmark {title}",
        title=dict(
            y=0.98,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=18),
        ),
        margin=dict(
            t=80,
            l=50,
            r=50,
            b=60,
        ),
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1,
            font=dict(size=14),
            tracegroupgap=5,
        ),
        font=dict(size=10),
        template="plotly_dark",
    )

    # Reduce subplot title font size
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=14)

    # Add a hover template, already shows framework_backend by default
    # customdata indices: [0]=batch_size, [1]=prompt_tps, [2]=generation_tps,
    # [3]=prompt_time_sec, [4]=generation_time_sec, [5]=total_time_sec,
    # [6]=peak_memory_gib, [7]=prompt_tps_std, [8]=generation_tps_std,
    # [9]=prompt_time_sec_std, [10]=generation_time_sec_std, [11]=peak_memory_gib_std
    fig.update_traces(
        hovertemplate=(
            "<b>Batch size:</b>              %{customdata[0]:>9.0f}<br>"
            "<b>Num prompt tokens:</b>       %{x:>9.0f}<br>"
            "<b>Prompt speed (tokens/s):</b> %{customdata[1]:>9.2f} ± %{customdata[7]:.2f}<br>"
            "<b>Gen. speed (tokens/s):</b>   %{customdata[2]:>9.2f} ± %{customdata[8]:.2f}<br>"
            "<b>Prompt time (s):</b>         %{customdata[3]:>9.4f} ± %{customdata[9]:.4f}<br>"
            "<b>Gen. time (s):</b>           %{customdata[4]:>9.4f} ± %{customdata[10]:.4f}<br>"
            "<b>Total time (s):</b>          %{customdata[5]:>9.4f}<br>"
            "<b>Peak memory (GiB):</b>       %{customdata[6]:>9.4f} ± %{customdata[11]:.4f}<br>"
        ),
        mode="markers",
    )
    fig.update_layout(
        hoverlabel=dict(
            font_family="Menlo, monospace",
            font_size=14,
        )
    )
    return fig
