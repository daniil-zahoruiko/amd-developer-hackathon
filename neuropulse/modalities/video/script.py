from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field

from modalities.video.params import VideoSceneParams


@dataclass
class VideoScene:
    id: int
    narration: str
    visual_description: str
    engagement_score: float | None = None
    params: VideoSceneParams = field(default_factory=VideoSceneParams.default)


@dataclass
class VideoScript:
    title: str
    scenes: list[VideoScene]

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "VideoScript":
        data = json.loads(s)
        scenes = []
        for raw in data.get("scenes", []):
            params_data = raw.get("params", {})
            params = VideoSceneParams(
                animation_speed=float(params_data.get("animation_speed", 1.0)),
                words_per_frame=int(params_data.get("words_per_frame", 8)),
                pause_after_sec=float(params_data.get("pause_after_sec", 1.0)),
                visual_elements=int(params_data.get("visual_elements", 2)),
                emphasis_words=list(params_data.get("emphasis_words", [])),
            )
            scenes.append(VideoScene(
                id=int(raw.get("id", len(scenes))),
                narration=str(raw.get("narration", "")),
                visual_description=str(raw.get("visual_description", "")),
                engagement_score=raw.get("engagement_score"),
                params=params,
            ))
        return cls(title=str(data.get("title", "")), scenes=scenes)

    def to_narration_text(self) -> str:
        return "\n\n".join(scene.narration for scene in self.scenes)

    def scene_durations_sec(self) -> list[float]:
        durations = []
        for scene in self.scenes:
            words = len(scene.narration.split())
            speaking_sec = words / 2.5
            total = speaking_sec + scene.params.pause_after_sec
            durations.append(total)
        return durations

    def with_scores(self, scores: dict[int, float]) -> "VideoScript":
        new_scenes = []
        for scene in self.scenes:
            new_scene = copy.deepcopy(scene)
            if scene.id in scores:
                new_scene.engagement_score = scores[scene.id]
            new_scenes.append(new_scene)
        return VideoScript(title=self.title, scenes=new_scenes)
