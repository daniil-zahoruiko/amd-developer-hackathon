# NeuroPulse

A text refinement tool that uses TRIBE v2 brain simulation as a scoring signal.
Paste or generate text, then iteratively refine it one step at a time. Edit freely between refinements — the working text is always whatever is in the textbox.

## Running

```bash
pip install -r requirements.txt
python app.py
```

Opens at http://localhost:7860

## What to implement to make it real

| File | What to implement |
|---|---|
| `scoring/tribe_runner.py` | `TRIBERunner.__init__()` — load model; `TRIBERunner.run()` — forward pass |
| `scoring/roi_extractor.py` | `ROIExtractor.extract_segment_scores()` — real ROI indices + engagement formula |
| `modalities/text/preprocessor.py` | `TextPreprocessor.preprocess()` — TRIBE v2 input format |
| `modalities/text/generator.py` | `TextGenerator.generate_from_topic()` and `.refine()` — LLM calls |
| `optimization/optimizer.py` | `Optimizer.suggest_refinement_instruction()` — weak-segment analysis |

`optimizer.run_one_iteration()` is already fully wired — implementing the stubs above is sufficient to make the real system work.

## Adding a new modality (audio, video)

Create `modalities/{name}/` with `params.py`, `generator.py`, `preprocessor.py`
following the same pattern as `modalities/text/`.
`scoring/` and `optimization/` require no changes — they are modality-agnostic.
