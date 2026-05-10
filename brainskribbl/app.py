import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
from starlette.datastructures import MutableHeaders
from starlette.middleware import Middleware

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
    """Score text without refinement: preprocess -> TRIBE -> ROI extraction.

    Returns (seg_scores, absolute_mean) where seg_scores is the (seq_ids, scores, seq_to_text)
    tuple for the timeline (relative, within-run) and absolute_mean is the cross-run
    comparable summary score.
    """
    preds, segments, df = tribe_runner.run_text(text)
    seq_ids, scores = roi_extractor.extract_segment_scores(preds, segments)
    _, abs_scores   = roi_extractor.extract_absolute_score(preds, segments)
    seq_to_text = _build_seq_to_text(df)
    return (seq_ids, scores, seq_to_text), float(abs_scores.mean())


def _log_rows(history: list[dict]) -> list[list]:
    return [[h["label"]] for h in history]


def handle_generate(topic: str, history: list[dict], step_count: int):
    yield (
        gr.update(), gr.update(), gr.update(), gr.update(), history, step_count,
        gr.update(), gr.update(),
        gr.update(value="⏳ Generating…", interactive=False),
    )
    try:
        if not topic.strip():
            raise gr.Error("Please enter a topic first.")
        text = generator.generate_from_topic(topic, params)
        seg_scores, abs_mean = _score_only(text)
        history = history + [{"label": "Draft"}]
        timeline_fig = build_timeline_chart(seg_scores)
        log = _log_rows(history)
        yield (
            text, f"{abs_mean:.1f}", timeline_fig, log, history, step_count,
            gr.update(visible=True), gr.update(visible=True),
            gr.update(value="Generate draft", interactive=True),
        )
    except gr.Error:
        yield (
            gr.update(), gr.update(), gr.update(), gr.update(), history, step_count,
            gr.update(), gr.update(),
            gr.update(value="Generate draft", interactive=True),
        )
        raise
    except Exception as e:
        yield (
            gr.update(), gr.update(), gr.update(), gr.update(), history, step_count,
            gr.update(), gr.update(),
            gr.update(value="Generate draft", interactive=True),
        )
        raise gr.Error(str(e))


def handle_score(current_text: str, history: list[dict]):
    yield (
        gr.update(), gr.update(), gr.update(), history,
        gr.update(), gr.update(),
        gr.update(value="⏳ Scoring…", interactive=False),
    )
    try:
        if not current_text.strip():
            raise gr.Error("Working text is empty. Generate or paste some text first.")
        seg_scores, abs_mean = _score_only(current_text)
        history = history + [{"label": "Score"}]
        timeline_fig = build_timeline_chart(seg_scores)
        log = _log_rows(history)
        yield (
            f"{abs_mean:.1f}", timeline_fig, log, history,
            gr.update(visible=True), gr.update(visible=True),
            gr.update(value="Score", interactive=True),
        )
    except gr.Error:
        yield (
            gr.update(), gr.update(), gr.update(), history,
            gr.update(), gr.update(),
            gr.update(value="Score", interactive=True),
        )
        raise
    except Exception as e:
        yield (
            gr.update(), gr.update(), gr.update(), history,
            gr.update(), gr.update(),
            gr.update(value="Score", interactive=True),
        )
        raise gr.Error(str(e))


def handle_audio_score(audio_file):
    yield (
        gr.update(), gr.update(),
        gr.update(value="⏳ Analyzing…", interactive=False),
    )
    try:
        if audio_file is None:
            raise gr.Error("Please upload an audio file first.")
        path = audio_file if isinstance(audio_file, str) else audio_file.name
        preds, segments, df = tribe_runner.run_audio(path)
        seq_ids, scores  = roi_extractor.extract_segment_scores(preds, segments)
        _, abs_scores    = roi_extractor.extract_absolute_score(preds, segments)
        seq_to_text = _build_seq_to_text(df)
        fig = build_timeline_chart((seq_ids, scores, seq_to_text))
        yield (
            fig, f"{float(abs_scores.mean()):.1f}",
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
        gr.update(), gr.update(), gr.update(), gr.update(), history, step_count, False,
        gr.update(), gr.update(),
        gr.update(value="⏳ Refining…", interactive=False),
    )
    try:
        if not current_text.strip():
            raise gr.Error("Working text is empty. Generate or paste some text first.")
        refined, seg_scores, abs_mean = optimizer.run_one_iteration(
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
            refined, f"{abs_mean:.1f}", timeline_fig, log, history, step_count, False,
            gr.update(visible=True), gr.update(visible=True),
            gr.update(value="✦ Refine", interactive=True),
        )
    except gr.Error:
        yield (
            gr.update(), gr.update(), gr.update(), gr.update(), history, step_count, False,
            gr.update(), gr.update(),
            gr.update(value="✦ Refine", interactive=True),
        )
        raise
    except Exception as e:
        yield (
            gr.update(), gr.update(), gr.update(), gr.update(), history, step_count, False,
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
                with gr.Column(scale=1):
                    text_mean = gr.Textbox(
                        label="Mean engagement score (0–100, cross-run comparable)",
                        interactive=False,
                        placeholder="—",
                    )
                with gr.Column(scale=4):
                    timeline_plot = gr.Plot(label="Engagement by segment (relative)")

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
                    text_mean,
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
                    text_mean,
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
                    text_mean,
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

        # ── Audio tab ─────────────────────────────────────────────────────────
        with gr.Tab("Audio"):
            gr.Markdown("Upload an audio track to score brain engagement using language, attention, and default-mode network signals.")

            audio_input = gr.File(
                label="Upload audio",
                file_types=[".wav", ".mp3", ".flac", ".ogg"],
                type="filepath",
            )
            audio_score_btn = gr.Button("Evaluate", variant="primary")

            with gr.Row() as audio_results_row:
                with gr.Column(scale=1):
                    audio_mean = gr.Textbox(
                        label="Mean engagement score (0–100, cross-run comparable)",
                        interactive=False,
                        placeholder="—",
                    )
                with gr.Column(scale=4):
                    audio_plot = gr.Plot(label="Engagement over time")

            audio_score_btn.click(
                fn=handle_audio_score,
                inputs=[audio_input],
                outputs=[audio_plot, audio_mean, audio_score_btn],
            )


class _NoBufferMiddleware:
    """Disable proxy buffering so Cloudflare tunnels don't swallow SSE chunks."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def patched_send(message):
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append("X-Accel-Buffering", "no")
            await send(message)

        await self.app(scope, receive, patched_send)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        app_kwargs={"middleware": [Middleware(_NoBufferMiddleware)]},
    )
