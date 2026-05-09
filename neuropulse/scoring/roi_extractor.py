import numpy as np
from nilearn import datasets

class ROIExtractor:
    def __init__(self):
        destrieux = datasets.fetch_atlas_surf_destrieux()
        ll = destrieux['map_left']
        lr = destrieux['map_right']
        names = destrieux['labels']

        def verts(label_names, hemispheres='both'):
            idx = []
            for name in label_names:
                i = names.index(name)
                if hemispheres in ('left', 'both'):
                    idx += list(np.where(ll == i)[0])
                if hemispheres in ('right', 'both'):
                    idx += list(np.where(lr == i)[0] + 10242)
            return np.array(idx)

        self.DAN = verts([
            'S_intrapariet_and_P_trans', 'G_parietal_sup', 'G_precentral'
        ])
        self.LANGUAGE = verts([
            'G_front_inf-Opercular', 'G_front_inf-Triangul',
            'G_temp_sup-G_T_transv', 'G_temp_sup-Lateral',
            'G_temp_sup-Plan_tempo', 'S_temporal_sup'
        ], hemispheres='left')   # language is left-lateralized
        self.DMN = verts([
            'G_and_S_frontomargin',
            'G_cingul-Post-dorsal',
            'G_cingul-Post-ventral',
            'S_subparietal',
            'G_precuneus',
            'G_pariet_inf-Angular',
        ])
        self.ACC = verts([
            'G_and_S_cingul-Mid-Ant',
            'G_and_S_cingul-Ant',
        ])
        # Ventral visual stream — fusiform, lingual, parahippocampal, inferior occipital
        self.VENTRAL = verts([
            'G_oc-temp_lat-fusifor',
            'G_oc-temp_med-Lingual',
            'G_oc-temp_med-Parahip',
            'G_and_S_occipital_inf',
        ])

        # Underscore-suffixed aliases used by the video pipeline's roi_indices dict
        self.VENTRAL_STREAM_VOXELS = self.VENTRAL
        self.LANGUAGE_VOXELS       = self.LANGUAGE
        self.DAN_VOXELS            = self.DAN
        self.DMN_VOXELS            = self.DMN
        self.ACC_VOXELS            = self.ACC

    def extract_segment_scores(self, preds, segments):
        def zscore(x: np.ndarray) -> np.ndarray:
            """
            Standardize to mean=0, std=1 across the array.
            Makes ROI weights meaningful by equalizing scales before combining —
            otherwise whichever ROI happens to have the largest dynamic range
            dominates the composite regardless of its assigned weight.
            Falls back to returning zeros if std is ~0 (flat signal).
            """
            std = x.std()
            if std < 1e-8:
                return np.zeros_like(x)
            return (x - x.mean()) / std


        # Step 1: For each preds row, find the dominant sequence_id
        # (the most recently started word in that segment = "current" sentence)
        row_sequence_ids = []

        for seg in segments:
            words = seg.events[seg.events.type == "Word"]
            if len(words) == 0:
                row_sequence_ids.append(None)
                continue
            # Most recently started word = the sentence being spoken right now
            current_word = words.loc[words["start"].idxmax()]
            row_sequence_ids.append(int(current_word["sequence_id"]))

        row_sequence_ids = np.array(
            [s if s is not None else -1 for s in row_sequence_ids]
        )

        # Step 2: Average preds for each sequence_id (collapses the 3× repetition)
        unique_seqs = sorted(s for s in set(row_sequence_ids) if s >= 0)

        seq_preds = {}
        for seq_id in unique_seqs:
            mask = row_sequence_ids == seq_id
            seq_preds[seq_id] = preds[mask].mean(axis=0)  # shape: (20484,)

        # Step 3: Stack into an array and score
        seq_ids_arr    = np.array(unique_seqs)
        preds_by_seq   = np.stack([seq_preds[s] for s in unique_seqs])  # (31, 20484)

        language_act = preds_by_seq[:, self.LANGUAGE].mean(axis=1)
        DAN_act      = preds_by_seq[:, self.DAN].mean(axis=1)
        DMN_act      = preds_by_seq[:, self.DMN].mean(axis=1)
        ACC_act      = preds_by_seq[:, self.ACC].mean(axis=1)

        raw_composite = (
            0.30 * zscore(language_act)
            + 0.30 * zscore(DAN_act)
            - 0.25 * zscore(DMN_act)
            + 0.15 * zscore(ACC_act)
        )
        # Sigmoid maps the z-score composite to (0, 100) on a fixed scale so
        # scores are comparable across different texts (no per-call min/max stretch).
        # raw_composite ≈ 0 → 50 (neutral); ±3 → ~95 / ~5 (strong engagement / disengagement).
        engagement_score = 100.0 / (1.0 + np.exp(-raw_composite))

        return seq_ids_arr, engagement_score

    def mean_score(self, engagement_scores: np.ndarray) -> float:
        """Returns mean engagement_score across sequences."""
        if len(engagement_scores) == 0:
            return 0.0
        return float(engagement_scores.mean())
