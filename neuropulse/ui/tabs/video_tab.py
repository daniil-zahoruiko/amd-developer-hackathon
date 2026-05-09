from __future__ import annotations

import tempfile

import gradio as gr
import numpy as np
import pandas as pd

from modalities.video.script import VideoScript
from scoring.video_scorer import VideoScorer
from ui.components.timeline_chart import build_history_chart, build_timeline_chart


def _map_preds_to_scenes(
    preds: np.ndarray,
    segments,
    scene_durations: list[float],
) -> np.ndarray:
    """
    Distribute TRIBE v2 prediction rows across scenes by wall-clock timing.

    Returns an array of shape (n_scenes, n_voxels) where each row is the
    mean activation for segments whose start time falls within that scene.
    Scenes with no matching segments get a zero vector.
    """
    cumulative = np.cumsum([0.0] + scene_durations)
    full_start = segments[0].start
    row_times = np.array([s.start - full_start for s in segments])

    scene_preds: list[np.ndarray] = []
    for i in range(len(scene_durations)):
        t_start = cumulative[i]
        t_end = cumulative[i + 1]
        mask = (row_times >= t_start) & (row_times < t_end)
        if mask.sum() > 0:
            scene_preds.append(preds[mask].mean(axis=0))
        else:
            scene_preds.append(np.zeros(preds.shape[1]))

    return np.stack(scene_preds)


def build_scene_instruction(row, df: pd.DataFrame) -> str:
    """
    Derive a targeted rewrite instruction for a low-scoring scene by comparing
    its per-ROI activations to the batch mean.
    """
    instructions: list[str] = []

    lang_mean    = df["language_raw"].mean()
    dan_mean     = df["DAN_raw"].mean()
    dmn_mean     = df["DMN_raw"].mean()
    ventral_mean = df["ventral_raw"].mean()

    if row.language_raw < lang_mean:
        instructions.append("simplify narration, use shorter sentences")
    if row.DAN_raw < dan_mean:
        instructions.append("add a concrete fact, name, or number")
    if row.DMN_raw > dmn_mean:
        instructions.append("add an unexpected angle or contrast")
    if row.ventral_raw < ventral_mean:
        instructions.append("enrich visual_description with more distinct imagery")

    if not instructions:
        instructions.append("improve overall engagement and clarity")

    return "; ".join(instructions)


def build_video_tab(
    config,
    model,
    generator,
    renderer,
    roi_extractor,
    scorer: VideoScorer,
    roi_indices: dict,
) -> None:
    """
    Build the Video tab UI inside the active gr.Blocks context.
    All event handlers are defined and wired here.
    """

    # ── Section 1: Generate ──────────────────────────────────────────────────
    with gr.Row():
        topic_box = gr.Textbox(
            label="Topic",
            placeholder="e.g. the science of sleep",
            lines=1,
            scale=3,
        )
        n_scenes_slider = gr.Slider(
            label="Scenes",
            minimum=3,
            maximum=12,
            step=1,
            value=6,
            scale=1,
        )
    generate_btn = gr.Button("Generate script", size="sm", variant="secondary")

    # ── Section 2: Script editor ─────────────────────────────────────────────
    script_editor = gr.Textbox(
        label="Script JSON — edit freely",
        lines=20,
        interactive=True,
        elem_id="video_script_editor",
    )
    gr.Markdown(
        "_Edit `narration` and `visual_description` in any scene, "
        "then click Render & score._"
    )

    # ── Section 3: Render ────────────────────────────────────────────────────
    with gr.Row():
        quality_radio = gr.Radio(
            choices=["low", "high"],
            value="low",
            label="Render quality",
        )
        render_btn = gr.Button("▶ Render & score", variant="primary")
    status_box = gr.Textbox(label="Status", interactive=False, lines=1)

    # ── Section 4 & 5: Results + Refine (hidden until first render) ──────────
    with gr.Group(visible=False) as results_group:
        video_player = gr.Video(label="Rendered video", interactive=False)
        with gr.Row():
            timeline_plot = gr.Plot(label="Engagement by scene")
            history_plot  = gr.Plot(label="Score history")

        # ── Section 5: Refine ────────────────────────────────────────────────
        with gr.Row():
            threshold_slider = gr.Slider(
                label="Rewrite scenes below score",
                minimum=0,
                maximum=100,
                step=5,
                value=40,
            )
            refine_btn = gr.Button("✦ Refine script", variant="primary")
        changes_box = gr.Textbox(label="Changes made", interactive=False, lines=3)

    # ── State ────────────────────────────────────────────────────────────────
    scores_state  = gr.State([])          # list[dict] — per-scene score rows
    history_state = gr.State([])          # list[float] — mean_score per iteration
    scorer_state  = gr.State(scorer)      # VideoScorer instance (holds baseline)

    # ── Event: Generate script ───────────────────────────────────────────────
    def handle_generate(topic: str, n_scenes: int):
        if not topic.strip():
            raise gr.Error("Please enter a topic.")
        script = generator.generate_script(topic, int(n_scenes))
        return script.to_json(), "Script generated — click Render & score"

    generate_btn.click(
        fn=handle_generate,
        inputs=[topic_box, n_scenes_slider],
        outputs=[script_editor, status_box],
    )

    # ── Event: Render & score ────────────────────────────────────────────────
    def handle_render_and_score(
        script_json: str,
        quality: str,
        scores_state_val: list,
        history_state_val: list,
        scorer_val: VideoScorer,
    ):
        if not script_json.strip():
            raise gr.Error("Script is empty. Generate a script first.")

        script = VideoScript.from_json(script_json)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = f.name

        out_path, scene_durations = renderer.render(script, out_path, quality)

        df_events = model.get_events_dataframe(video_path=out_path)
        preds, segments = model.predict(events=df_events)

        preds_by_scene = _map_preds_to_scenes(preds, segments, scene_durations)

        df_scores = scorer_val.score(preds_by_scene, roi_indices)
        df_scores["scene_id"] = range(len(script.scenes))

        mean_eng = float(df_scores["mean_score"].mean())
        history  = history_state_val + [mean_eng]

        # Build timeline chart using existing tuple API.
        scene_ids  = df_scores["scene_id"].values
        eng_scores = df_scores["engagement_score"].values
        seq_to_text = {
            int(r.scene_id): script.scenes[int(r.scene_id)].narration
            for _, r in df_scores.iterrows()
        }
        timeline_fig = build_timeline_chart((scene_ids, eng_scores, seq_to_text))

        history_fig = build_history_chart(
            [{"label": f"v{i}", "score": s} for i, s in enumerate(history)]
        )

        updated_script = script.with_scores(
            {int(r.scene_id): r.engagement_score for _, r in df_scores.iterrows()}
        )

        return (
            out_path,
            timeline_fig,
            history_fig,
            df_scores.to_dict("records"),
            history,
            gr.update(visible=True),
            f"Scored {len(script.scenes)} scenes — mean engagement {mean_eng:.3f}",
            updated_script.to_json(),
        )

    render_outputs = [
        video_player,
        timeline_plot,
        history_plot,
        scores_state,
        history_state,
        results_group,
        status_box,
        script_editor,
    ]

    render_btn.click(
        fn=handle_render_and_score,
        inputs=[script_editor, quality_radio, scores_state, history_state, scorer_state],
        outputs=render_outputs,
    )

    # ── Event: Refine script (chains into Render & score) ────────────────────
    def handle_refine(
        script_json: str,
        threshold: float,
        scores_state_val: list,
        scorer_val: VideoScorer,
    ):
        if not script_json.strip():
            raise gr.Error("Script is empty. Generate and render a script first.")

        script = VideoScript.from_json(script_json)
        df = pd.DataFrame(scores_state_val)

        if df.empty:
            raise gr.Error("No scores available. Click Render & score first.")

        weak = df[df["engagement_score"] < threshold]
        if weak.empty:
            return script_json, "All scenes above threshold — no changes made."

        instructions = {
            int(r.scene_id): build_scene_instruction(r, df)
            for _, r in weak.iterrows()
        }

        refined_script = generator.rewrite_scenes(
            script, list(instructions.keys()), instructions
        )

        changes = "\n".join(
            f"Scene {sid}: {instr}" for sid, instr in instructions.items()
        )
        return refined_script.to_json(), changes

    refine_btn.click(
        fn=handle_refine,
        inputs=[script_editor, threshold_slider, scores_state, scorer_state],
        outputs=[script_editor, changes_box],
    ).then(
        fn=handle_render_and_score,
        inputs=[script_editor, quality_radio, scores_state, history_state, scorer_state],
        outputs=render_outputs,
    )
