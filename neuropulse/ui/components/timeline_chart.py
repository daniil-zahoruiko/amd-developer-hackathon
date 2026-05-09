import plotly.graph_objects as go


def build_timeline_chart(seq_scores: tuple) -> go.Figure:
    """Bar chart showing engagement score per sequence."""
    seq_ids, scores, seq_to_text = seq_scores

    MAX_LABEL = 38
    short_labels = []
    full_texts = []
    for sid in seq_ids:
        full = seq_to_text.get(int(sid), f"Seq {sid}")
        short = (full[:MAX_LABEL] + "…") if len(full) > MAX_LABEL else full
        short_labels.append(short)
        full_texts.append(full)

    colors = []
    for score in scores:
        if score >= 65:
            colors.append("#5DCAA5")
        elif score >= 40:
            colors.append("#EF9F27")
        else:
            colors.append("#E24B4A")

    fig = go.Figure(
        go.Bar(
            x=short_labels,
            y=scores,
            marker_color=colors,
            customdata=full_texts,
            hovertemplate="%{customdata}<br><b>Score: %{y:.1f}</b><extra></extra>",
        )
    )
    fig.update_layout(
        title="Engagement by sentence",
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        xaxis=dict(showgrid=False, tickangle=-35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=40, b=160),
        font=dict(size=11),
    )
    return fig


def build_history_chart(history: list[dict]) -> go.Figure:
    """Line + marker chart showing mean engagement score across refinement steps."""
    labels = [h["label"] for h in history]
    scores = [h["score"] for h in history]

    annotations = []
    if history:
        for idx in [0, len(history) - 1]:
            if idx < len(history):
                annotations.append(
                    dict(
                        x=labels[idx],
                        y=scores[idx],
                        text=f"{scores[idx]:.1f}",
                        showarrow=False,
                        yshift=12,
                        font=dict(size=11),
                    )
                )

    fig = go.Figure(
        go.Scatter(
            x=labels,
            y=scores,
            mode="lines+markers",
            line=dict(color="#378ADD", width=2),
            marker=dict(color="#378ADD", size=8, symbol="circle"),
            hovertemplate="%{x}: %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Score across refinements",
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        xaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=annotations,
        margin=dict(l=40, r=20, t=40, b=60),
        font=dict(size=12),
    )
    return fig
