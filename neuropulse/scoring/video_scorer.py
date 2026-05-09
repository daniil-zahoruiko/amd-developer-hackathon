import numpy as np
import pandas as pd

_ROI_ORDER = ["ventral", "language", "DAN", "DMN", "ACC"]


class VideoScorer:
    def __init__(self):
        self._baseline_means: np.ndarray | None = None
        self._baseline_stds: np.ndarray | None = None

    def score(
        self,
        preds_by_scene: np.ndarray,
        roi_indices: dict,
    ) -> pd.DataFrame:
        """
        Score each scene's brain predictions.

        preds_by_scene: shape (n_scenes, 20484)
        roi_indices:    {"ventral": array, "language": array, "DAN": array,
                         "DMN": array, "ACC": array}

        Returns DataFrame with columns:
            scene_id, engagement_score, mean_score,
            ventral_raw, language_raw, DAN_raw, DMN_raw, ACC_raw
        """
        ventral_act  = preds_by_scene[:, roi_indices["ventral"]].mean(axis=1)
        language_act = preds_by_scene[:, roi_indices["language"]].mean(axis=1)
        dan_act      = preds_by_scene[:, roi_indices["DAN"]].mean(axis=1)
        dmn_act      = preds_by_scene[:, roi_indices["DMN"]].mean(axis=1)
        acc_act      = preds_by_scene[:, roi_indices["ACC"]].mean(axis=1)

        raws = np.stack([ventral_act, language_act, dan_act, dmn_act, acc_act])

        # Set baseline on first call and keep it fixed so scores are
        # comparable across refinement iterations within a session.
        if self._baseline_means is None:
            self._baseline_means = raws.mean(axis=1)
            self._baseline_stds = np.where(
                raws.std(axis=1) < 1e-8, 1.0, raws.std(axis=1)
            )

        def zref(arr: np.ndarray, name: str) -> np.ndarray:
            i = _ROI_ORDER.index(name)
            return (arr - self._baseline_means[i]) / self._baseline_stds[i]

        raw_composite = (
            0.25 * zref(ventral_act,  "ventral")
            + 0.20 * zref(language_act, "language")
            + 0.25 * zref(dan_act,      "DAN")
            - 0.20 * zref(dmn_act,      "DMN")
            + 0.10 * zref(acc_act,      "ACC")
        )

        engagement_score = 100.0 / (1.0 + np.exp(-raw_composite))
        mean_score = float(engagement_score.mean())

        rows = [
            {
                "scene_id":        i,
                "engagement_score": float(engagement_score[i]),
                "mean_score":       mean_score,
                "ventral_raw":     float(ventral_act[i]),
                "language_raw":    float(language_act[i]),
                "DAN_raw":         float(dan_act[i]),
                "DMN_raw":         float(dmn_act[i]),
                "ACC_raw":         float(acc_act[i]),
            }
            for i in range(len(preds_by_scene))
        ]
        return pd.DataFrame(rows)
