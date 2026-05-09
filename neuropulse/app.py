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
from ui.components.timeline_chart import build_timeline_chart

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
    try:
        if not topic.strip():
            raise gr.Error("Please enter a topic first.")
        text = generator.generate_from_topic(topic, params)
        seg_scores = _score_only(text)
        history = history + [{"label": "Draft"}]
        timeline_fig = build_timeline_chart(seg_scores)
        log = _log_rows(history)
        return (
            text,
            timeline_fig,
            log,
            history,
            step_count,
            gr.update(visible=True),
            gr.update(visible=True),
        )
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(str(e))


def handle_score(current_text: str, history: list[dict]):
    try:
        if not current_text.strip():
            raise gr.Error("Working text is empty. Generate or paste some text first.")
        seg_scores = _score_only(current_text)
        history = history + [{"label": "Score"}]
        timeline_fig = build_timeline_chart(seg_scores)
        log = _log_rows(history)
        return (
            timeline_fig,
            log,
            history,
            gr.update(visible=True),
            gr.update(visible=True),
        )
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(str(e))


def handle_refine(
    current_text: str,
    history: list[dict],
    step_count: int,
    user_edited: bool,
):
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
        return (
            refined,
            timeline_fig,
            log,
            history,
            step_count,
            False,                   # reset user_edited_state
            gr.update(visible=True),
            gr.update(visible=True),
        )
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(str(e))


# --- UI ---
with gr.Blocks(title=config.APP_TITLE) as demo:
    # Header
    gr.Markdown(f"# {config.APP_TITLE}")
    gr.Markdown(config.APP_DESCRIPTION)

    # Section 1 — Start
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

    # Section 2 — Working text
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

    # Section 3 — Charts
    with gr.Row(visible=False) as charts_row:
        timeline_plot = gr.Plot(label="Engagement by segment")

    # Section 4 — Step log
    with gr.Row(visible=False) as log_row:
        log_df = gr.Dataframe(
            headers=["Step"],
            label="Refinement log",
            interactive=False,
        )

    # State
    history_state = gr.State([])
    step_count_state = gr.State(0)
    user_edited_state = gr.State(False)

    # Event wiring
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
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
