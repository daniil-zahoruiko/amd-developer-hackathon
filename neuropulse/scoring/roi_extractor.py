import numpy as np


class ROIExtractor:
    def __init__(self):
        # MOCK voxel index ranges — replace with real indices from the TRIBE v2
        # paper or the NSD dataset atlas once you have the model weights.
        self.VENTRAL_STREAM_VOXELS: list[int] = list(range(0, 10000))
        self.LANGUAGE_VOXELS: list[int] = list(range(10000, 20000))
        self.PFC_VOXELS: list[int] = list(range(20000, 30000))

    def extract_segment_scores(self, activations: np.ndarray) -> list[dict]:
        """
        TODO: Implement this once TRIBE v2 is integrated.
        `activations` is shape (n_segments, 70000).

        For each segment:
          - Slice the relevant voxel indices using self.VENTRAL_STREAM_VOXELS,
            self.LANGUAGE_VOXELS, and self.PFC_VOXELS.
          - Compute per-ROI mean activations (or another summary statistic).
          - Combine them into a single engagement_score in [0, 100].
            E.g. a weighted sum: 0.4*language + 0.4*ventral + 0.2*pfc, rescaled.
          - Return a list of dicts with at minimum:
              {"segment_idx": int, "engagement_score": float}
            You may add any additional keys (e.g. language_score, pfc_load).

        The optimizer reads only "engagement_score" from each dict.
        """
        # TODO: replace with real ROI extraction
        n = activations.shape[0]
        return [
            {"segment_idx": i, "engagement_score": float(np.random.uniform(30, 80))}
            for i in range(n)
        ]

    def mean_score(self, segment_scores: list[dict]) -> float:
        """Returns mean engagement_score across segments."""
        if not segment_scores:
            return 0.0
        return sum(s["engagement_score"] for s in segment_scores) / len(segment_scores)
