import json
import os
import shutil
import subprocess
import tempfile

import torch
from diffusers import StableDiffusionXLPipeline
from gtts import gTTS
from PIL import Image

from modalities.video.script import VideoScene, VideoScript

_RESOLUTION = {"low": (854, 480), "high": (1920, 1080)}
_FPS = 24
_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# How many zoom-speed steps per frame, scaled by animation_speed param.
_BASE_ZOOM_DELTA = 0.0010


class VideoRenderer:
    def __init__(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            _SDXL_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")  # ROCm exposes itself as a CUDA device
        self.pipe.enable_attention_slicing()

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        script: VideoScript,
        output_path: str,
        quality: str = "low",
    ) -> tuple[str, list[float]]:
        width, height = _RESOLUTION.get(quality, _RESOLUTION["low"])
        tmpdir = tempfile.mkdtemp(prefix="neuropulse_")

        try:
            scene_mp4s: list[str] = []
            for scene in script.scenes:
                mp4 = os.path.join(tmpdir, f"scene_{scene.id}.mp4")
                self._render_scene(scene, mp4, width, height)
                scene_mp4s.append(mp4)

            scene_durations = [self._get_duration(p) for p in scene_mp4s]

            concat_txt = os.path.join(tmpdir, "concat.txt")
            with open(concat_txt, "w", encoding="utf-8") as f:
                for p in scene_mp4s:
                    f.write(f"file '{p}'\n")

            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_txt,
                    "-c", "copy",
                    output_path,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr}")

            return output_path, scene_durations

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _generate_image(self, prompt: str) -> Image.Image:
        negative = (
            "text, watermark, logo, caption, subtitle, words, letters, blurry, "
            "low quality, duplicate, deformed, ugly, bad anatomy"
        )
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=1024,
            height=1024,
            num_inference_steps=25,
            guidance_scale=7.5,
        )
        return result.images[0]

    def _render_scene(
        self,
        scene: VideoScene,
        output_path: str,
        width: int,
        height: int,
    ) -> None:
        tmpdir = tempfile.mkdtemp(prefix="neuropulse_scene_")
        try:
            # ── SDXL image ────────────────────────────────────────────────────
            img = self._generate_image(scene.visual_description)
            # Scale to 1.4× output to give zoompan headroom for all zoom levels
            scaled_w = int(width * 1.4)
            scaled_h = int(height * 1.4)
            img = img.resize((scaled_w, scaled_h), Image.LANCZOS)
            img_path = os.path.join(tmpdir, "scene.png")
            img.save(img_path)

            # ── Narration audio ───────────────────────────────────────────────
            audio_path = os.path.join(tmpdir, "narration.mp3")
            tts = gTTS(text=scene.narration.strip() or ".", lang="en", slow=False)
            tts.save(audio_path)
            audio_dur = self._get_duration(audio_path)
            total_dur = audio_dur + scene.params.pause_after_sec

            # ── Ken Burns filter ──────────────────────────────────────────────
            n_frames = max(1, int(total_dur * _FPS))
            zoom_delta = _BASE_ZOOM_DELTA * scene.params.animation_speed
            kb_filter = _ken_burns_filter(
                scene.id, zoom_delta, n_frames, width, height
            )

            # Single-pass: loop image → Ken Burns video + padded audio → MP4
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", img_path,
                    "-i", audio_path,
                    "-filter_complex",
                    f"[0:v]{kb_filter}[v];[1:a]apad[aout]",
                    "-map", "[v]",
                    "-map", "[aout]",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-pix_fmt", "yuv420p",
                    "-preset", "ultrafast",
                    "-t", str(total_dur),
                    output_path,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg Ken Burns failed for scene {scene.id}:\n{result.stderr}"
                )

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _get_duration(self, media_path: str) -> float:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                media_path,
            ],
            capture_output=True,
            text=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])


def _ken_burns_filter(
    scene_id: int,
    zoom_delta: float,
    n_frames: int,
    width: int,
    height: int,
) -> str:
    """
    Return an ffmpeg zoompan filter string for a Ken Burns effect.

    Four styles cycle by scene_id:
      0 — slow zoom in,  centred
      1 — slow zoom out, centred
      2 — zoom in while panning right
      3 — zoom in while panning left
    """
    style = scene_id % 4

    if style == 0:  # zoom in, centred
        z = f"min(zoom+{zoom_delta:.4f},1.5)"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif style == 1:  # zoom out, centred
        z = f"if(eq(on,1),1.5,max(zoom-{zoom_delta:.4f},1.0))"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif style == 2:  # zoom in + pan right
        z = f"min(zoom+{zoom_delta:.4f},1.4)"
        x = f"min(x+0.4,iw-iw/zoom)"
        y = "ih/2-(ih/zoom/2)"
    else:            # zoom in + pan left
        z = f"min(zoom+{zoom_delta:.4f},1.4)"
        x = "max(x-0.4,0)"
        y = "ih/2-(ih/zoom/2)"

    return (
        f"zoompan=z='{z}':x='{x}':y='{y}'"
        f":d={n_frames}:s={width}x{height}:fps={_FPS}"
    )
