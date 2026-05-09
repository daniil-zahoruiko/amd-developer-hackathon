import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modalities.video.script import VideoScript


def _extract_json_str(text: str) -> str:
    """
    Pull the first complete JSON object out of raw LLM output.
    Handles markdown fences, leading prose, and trailing text.
    """
    text = text.strip()
    # Strip ```json ... ``` fences
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if fence:
        return fence.group(1).strip()
    # Walk forward to the first '{', then track brace depth
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM output: {text[:300]}")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Unbalanced braces in LLM output: {text[:300]}")


class VideoGenerator:
    MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

    _SYSTEM_GENERATE = (
        "You are a video script engine. Output ONLY valid JSON, no markdown fences, "
        "no explanation, no reasoning.\n\n"
        "Each scene's visual_description is a Stable Diffusion XL image prompt: "
        "rich, vivid, photorealistic or cinematic, describing lighting, mood, "
        "subject, and setting in detail. Do NOT mention text, words, or overlays.\n\n"
        "Schema:\n"
        '{\n  "title": "string",\n  "scenes": [\n    {\n'
        '      "id": 0,\n      "narration": "string",\n'
        '      "visual_description": "detailed SDXL image prompt string",\n'
        '      "params": {\n        "animation_speed": 1.0,\n'
        '        "words_per_frame": 8,\n        "pause_after_sec": 1.0,\n'
        '        "visual_elements": 2,\n        "emphasis_words": ["word1"]\n'
        "      }\n    }\n  ]\n}"
    )

    _SYSTEM_REWRITE = (
        "You are editing a video script. Rewrite ONLY the scenes listed. "
        "Return the COMPLETE script JSON with those scenes updated. "
        "Preserve all other scenes exactly. Only change narration, "
        "visual_description, and emphasis_words — do not change id or params structure. "
        "visual_description must be a rich Stable Diffusion XL image prompt: "
        "cinematic, detailed, describing lighting, mood, subject, and setting. "
        "Output ONLY valid JSON, no markdown fences, no explanation."
    )

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    def _generate(self, system: str, user: str, max_new_tokens: int = 2000) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def _parse_script(self, raw: str) -> VideoScript:
        return VideoScript.from_json(_extract_json_str(raw))

    def generate_script(self, topic: str, n_scenes: int = 6) -> VideoScript:
        user = f"Create a {n_scenes}-scene video script about: {topic}"
        raw = self._generate(self._SYSTEM_GENERATE, user)
        try:
            return self._parse_script(raw)
        except Exception as first_err:
            correction = (
                f"Your previous response failed to parse: {first_err}\n"
                f"Output was:\n{raw}\n\n"
                f"Return ONLY valid JSON matching the schema. "
                f"Original request: {user}"
            )
            raw2 = self._generate(self._SYSTEM_GENERATE, correction)
            try:
                return self._parse_script(raw2)
            except Exception as second_err:
                raise ValueError(
                    f"Model failed to produce valid script JSON twice. "
                    f"Last error: {second_err}\nLast output:\n{raw2}"
                )

    def rewrite_scenes(
        self,
        script: VideoScript,
        weak_scene_ids: list[int],
        instructions: dict[int, str],
    ) -> VideoScript:
        scene_lines = "\n".join(
            f"[scene {sid}] {instructions[sid]}" for sid in weak_scene_ids
        )
        user = (
            f"Here is the full script:\n{script.to_json()}\n\n"
            f"Rewrite these scenes:\n{scene_lines}"
        )
        raw = self._generate(self._SYSTEM_REWRITE, user, max_new_tokens=2500)
        try:
            refined = self._parse_script(raw)
        except Exception as err:
            correction = (
                f"Parse failed: {err}\nYour output:\n{raw}\n\n"
                f"Return ONLY valid JSON, no markdown, no explanation."
            )
            raw2 = self._generate(self._SYSTEM_REWRITE, correction)
            refined = self._parse_script(raw2)

        # Preserve engagement_scores for scenes that were not rewritten
        preserved = {s.id: s.engagement_score for s in script.scenes}
        rewritten_ids = set(weak_scene_ids)
        for scene in refined.scenes:
            if scene.id not in rewritten_ids and scene.id in preserved:
                scene.engagement_score = preserved[scene.id]

        return refined
