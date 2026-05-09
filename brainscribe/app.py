import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr

from config import Config
from modalities.text.params import TextParams
from modalities.text.generator import TextGenerator
from scoring.tribe_runner import TRIBERunner
from scoring.roi_extractor import ROIExtractor
from optimization.optimizer import Optimizer
from ui.components.timeline_chart import build_timeline_chart, build_video_chart

# --- Instantiation ---
config = Config()
params = TextParams.default()
tribe_runner = TRIBERunner(config)
roi_extractor = ROIExtractor()
generator = TextGenerator()
optimizer = Optimizer(generator)


def _build_seq_to_text(df) -> dict:
    words = df[df["type"] == "Word"]
    def _seq_text(group):
        lst = list(group["text"])
        n = len(lst)
        for period in range(1, n + 1):
            if n % period == 0 and lst[:period] * (n // period) == lst:
                return " ".join(lst[:period])
        return " ".join(lst)
    return words.groupby("sequence_id").apply(_seq_text).to_dict()


def _score_only(text: str) -> tuple:
    """Score text without refinement: preprocess -> TRIBE -> ROI extraction."""
    preds, segments, df = tribe_runner.run_text(text)
    seq_ids, scores = roi_extractor.extract_segment_scores(preds, segments)
    seq_to_text = _build_seq_to_text(df)
    return (seq_ids, scores, seq_to_text)


def _log_rows(history: list[dict]) -> list[list]:
    return [[h["label"]] for h in history]


def handle_generate(topic: str, history: list[dict], step_count: int):
    yield (
        gr.update(), gr.update(), gr.update(), history, step_count,
        gr.update(), gr.update(),
        gr.update(value="⏳ Generating…", interactive=False),
    )
    try:
        if not topic.strip():
            raise gr.Error("Please enter a topic first.")
        text = generator.generate_from_topic(topic, params)
        seg_scores = _score_only(text)
        history = history + [{"label": "Draft"}]
        timeline_fig = build_timeline_chart(seg_scores)
        log = _log_rows(history)
        yield (
            text, timeline_fig, log, history, step_count,
            gr.update(visible=True), gr.update(visible=True),
            gr.update(value="Generate draft", interactive=True),
        )
    except gr.Error:
        yield (
            gr.update(), gr.update(), gr.update(), history, step_count,
            gr.update(), gr.update(),
            gr.update(value="Generate draft", interactive=True),
        )
        raise
    except Exception as e:
        yield (
            gr.update(), gr.update(), gr.update(), history, step_count,
            gr.update(), gr.update(),
            gr.update(value="Generate draft", interactive=True),
        )
        raise gr.Error(str(e))


def handle_score(current_text: str, history: list[dict]):
    yield (
        gr.update(), gr.update(), history,
        gr.update(), gr.update(),
        gr.update(value="⏳ Scoring…", interactive=False),
    )
    try:
        if not current_text.strip():
            raise gr.Error("Working text is empty. Generate or paste some text first.")
        seg_scores = _score_only(current_text)
        history = history + [{"label": "Score"}]
        timeline_fig = build_timeline_chart(seg_scores)
        log = _log_rows(history)
        yield (
            timeline_fig, log, history,
            gr.update(visible=True), gr.update(visible=True),
            gr.update(value="Score", interactive=True),
        )
    except gr.Error:
        yield (
            gr.update(), gr.update(), history,
            gr.update(), gr.update(),
            gr.update(value="Score", interactive=True),
        )
        raise
    except Exception as e:
        yield (
            gr.update(), gr.update(), history,
            gr.update(), gr.update(),
            gr.update(value="Score", interactive=True),
        )
        raise gr.Error(str(e))


def handle_video_score(video_file):
    yield (
        gr.update(), gr.update(),
        gr.update(value="⏳ Analyzing…", interactive=False),
    )
    try:
        if video_file is None:
            raise gr.Error("Please upload a video first.")
        path = video_file if isinstance(video_file, str) else video_file.name
        preds, segments = tribe_runner.run_video(path)
        scores = roi_extractor.extract_video_scores(preds)
        mean_score = float(scores.mean())
        fig = build_video_chart(scores, segments)
        yield (
            fig, f"{mean_score:.1f}",
            gr.update(value="Evaluate", interactive=True),
        )
    except gr.Error:
        yield (
            gr.update(), gr.update(),
            gr.update(value="Evaluate", interactive=True),
        )
        raise
    except Exception as e:
        yield (
            gr.update(), gr.update(),
            gr.update(value="Evaluate", interactive=True),
        )
        raise gr.Error(str(e))


def handle_refine(
    current_text: str,
    history: list[dict],
    step_count: int,
    user_edited: bool,
):
    yield (
        gr.update(), gr.update(), gr.update(), history, step_count, False,
        gr.update(), gr.update(),
        gr.update(value="⏳ Refining…", interactive=False),
    )
    try:
        if not current_text.strip():
            raise gr.Error("Working text is empty. Generate or paste some text first.")
        refined, seg_scores = optimizer.run_one_iteration(
            current_text, params, tribe_runner, roi_extractor
        )
        step_count = step_count + 1
        label = (
            f"Human edit + Refinement {step_count}"
            if user_edited
            else f"Refinement {step_count}"
        )
        history = history + [{"label": label}]
        timeline_fig = build_timeline_chart(seg_scores)
        log = _log_rows(history)
        yield (
            refined, timeline_fig, log, history, step_count, False,
            gr.update(visible=True), gr.update(visible=True),
            gr.update(value="✦ Refine", interactive=True),
        )
    except gr.Error:
        yield (
            gr.update(), gr.update(), gr.update(), history, step_count, False,
            gr.update(), gr.update(),
            gr.update(value="✦ Refine", interactive=True),
        )
        raise
    except Exception as e:
        yield (
            gr.update(), gr.update(), gr.update(), history, step_count, False,
            gr.update(), gr.update(),
            gr.update(value="✦ Refine", interactive=True),
        )
        raise gr.Error(str(e))


# --- UI ---
with gr.Blocks(title=config.APP_TITLE) as demo:
    gr.Markdown(f"# {config.APP_TITLE}")
    gr.Markdown(config.APP_DESCRIPTION)

    with gr.Tabs():
        # ── Text tab ─────────────────────────────────────────────────────────
        with gr.Tab("Text"):
            with gr.Row():
                with gr.Column(scale=3):
                    topic_input = gr.Textbox(
                        label="Generate from topic",
                        placeholder="e.g. benefits of cold exposure",
                        lines=1,
                        elem_id="topic_input",
                    )
                    generate_btn = gr.Button("Generate draft", size="sm", variant="secondary")
                with gr.Column(scale=1):
                    gr.Markdown("**or paste your text directly below**")

            working_text = gr.Textbox(
                label="Working text",
                placeholder="Your text appears here. Edit freely at any time.",
                lines=14,
                interactive=True,
                elem_id="working_text",
            )
            with gr.Row():
                score_btn = gr.Button("Score", variant="secondary", size="sm")
                refine_btn = gr.Button("✦ Refine", variant="primary")

            with gr.Row(visible=False) as charts_row:
                timeline_plot = gr.Plot(label="Engagement by segment")

            with gr.Row(visible=False) as log_row:
                log_df = gr.Dataframe(
                    headers=["Step"],
                    label="Refinement log",
                    interactive=False,
                )

            history_state = gr.State([])
            step_count_state = gr.State(0)
            user_edited_state = gr.State(False)

            generate_btn.click(
                fn=handle_generate,
                inputs=[topic_input, history_state, step_count_state],
                outputs=[
                    working_text,
                    timeline_plot,
                    log_df,
                    history_state,
                    step_count_state,
                    charts_row,
                    log_row,
                    generate_btn,
                ],
            )

            working_text.change(
                fn=lambda _: True,
                inputs=[working_text],
                outputs=[user_edited_state],
            )

            score_btn.click(
                fn=handle_score,
                inputs=[working_text, history_state],
                outputs=[
                    timeline_plot,
                    log_df,
                    history_state,
                    charts_row,
                    log_row,
                    score_btn,
                ],
            )

            refine_btn.click(
                fn=handle_refine,
                inputs=[working_text, history_state, step_count_state, user_edited_state],
                outputs=[
                    working_text,
                    timeline_plot,
                    log_df,
                    history_state,
                    step_count_state,
                    user_edited_state,
                    charts_row,
                    log_row,
                    refine_btn,
                ],
            )

        # ── Video tab ─────────────────────────────────────────────────────────
        with gr.Tab("Video"):
            gr.Markdown("Upload a video to score brain engagement using visual, language, attention, and default-mode network signals.")

            video_input = gr.File(
                label="Upload video",
                file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                type="filepath",
            )
            video_score_btn = gr.Button("Evaluate", variant="primary")

            with gr.Row() as video_results_row:
                with gr.Column(scale=1):
                    video_mean = gr.Textbox(
                        label="Mean engagement score (0–100)",
                        interactive=False,
                        placeholder="—",
                    )
                with gr.Column(scale=4):
                    video_plot = gr.Plot(label="Engagement over time")

            video_score_btn.click(
                fn=handle_video_score,
                inputs=[video_input],
                outputs=[video_plot, video_mean, video_score_btn],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
