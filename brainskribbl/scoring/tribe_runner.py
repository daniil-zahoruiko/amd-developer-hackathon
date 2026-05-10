import numpy as np
import time
from tribev2 import TribeModel
import tempfile
from pathlib import Path


class TRIBERunner:
    def __init__(self, config):
        self.config = config
        self.model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

    def run_text(self, text: str):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            df = self.model.get_events_dataframe(text_path=tmp_path)
            preds, segments = self.model.predict(events=df)
            return preds, segments, df

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def run_audio(self, audio_path: str):
        df = self.model.get_events_dataframe(audio_path=audio_path)
        preds, segments = self.model.predict(events=df)
        return preds, segments, df