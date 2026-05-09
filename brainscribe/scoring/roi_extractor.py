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
        # Ventral visual stream: fusiform, lingual, parahippocampal, inferior temporal, inferior occipital
        self.ventral = verts([
            'G_oc-temp_lat-fusifor',
            'G_oc-temp_med-Lingual',
            'G_oc-temp_med-Parahip',
            'G_temporal_inf',
            'G_and_S_occipital_inf',
        ])

    def _aggregate_by_sequence(self, preds, segments):
        """Average preds over segments that share a sequence_id (collapses repetitions).

        Returns (seq_ids_arr, preds_by_seq) where preds_by_seq has shape (n_seqs, n_verts).
        """
        row_sequence_ids = []
        for seg in segments:
            words = seg.events[seg.events.type == "Word"]
            if len(words) == 0:
                row_sequence_ids.append(None)
                continue
            current_word = words.loc[words["start"].idxmax()]
            row_sequence_ids.append(int(current_word["sequence_id"]))

        row_sequence_ids = np.array(
            [s if s is not None else -1 for s in row_sequence_ids]
        )

        unique_seqs = sorted(s for s in set(row_sequence_ids) if s >= 0)
        seq_preds = {}
        for seq_id in unique_seqs:
            mask = row_sequence_ids == seq_id
            seq_preds[seq_id] = preds[mask].mean(axis=0)

        seq_ids_arr  = np.array(unique_seqs)
        preds_by_seq = np.stack([seq_preds[s] for s in unique_seqs])
        return seq_ids_arr, preds_by_seq

    def extract_segment_scores(self, preds, segments):
        """Relative engagement scores (z-scored within session).

        Good for within-run comparison: highlights which sentences are stronger
        or weaker relative to the rest of the text. Not comparable across runs.
        """
        def zscore(x: np.ndarray) -> np.ndarray:
            std = x.std()
            if std < 1e-8:
                return np.zeros_like(x)
            return (x - x.mean()) / std

        seq_ids_arr, preds_by_seq = self._aggregate_by_sequence(preds, segments)

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
        # Sigmoid maps to (0, 100). raw_composite ≈ 0 → 50; ±3 → ~95 / ~5.
        engagement_score = 100.0 / (1.0 + np.exp(-raw_composite))

        return seq_ids_arr, engagement_score

    def extract_absolute_score(self, preds, segments):
        """Absolute engagement scores comparable across different runs.

        Uses the ventral-anchored Task-Positive Network Index (TPNI):

          baseline = mean ventral visual activation across the run
                     (task-irrelevant region; corrects for run-level signal offset)

          TPN = 0.40 * (language − baseline)
              + 0.40 * (DAN − baseline)
              + 0.20 * (ACC − baseline)

          TNN = DMN − baseline

          TPNI = (TPN − TNN) / (|TPN| + |TNN| + ε)   ∈ (−1, 1)
          score = 50 × (1 + TPNI)                      ∈ (0, 100)

        The ratio form makes scores scale-invariant: the same sentence always
        receives the same score regardless of what other sentences appear in
        the same batch. 50 = neutral; >50 = task-positive networks dominate
        (engaged); <50 = DMN dominates (mind-wandering / disengagement).
        """
        seq_ids_arr, preds_by_seq = self._aggregate_by_sequence(preds, segments)

        language_act = preds_by_seq[:, self.LANGUAGE].mean(axis=1)
        DAN_act      = preds_by_seq[:, self.DAN].mean(axis=1)
        DMN_act      = preds_by_seq[:, self.DMN].mean(axis=1)
        ACC_act      = preds_by_seq[:, self.ACC].mean(axis=1)
        ventral_act  = preds_by_seq[:, self.ventral].mean(axis=1)

        # Ventral visual cortex: task-irrelevant during language/audio tasks.
        # Its mean provides a run-level offset correction without contaminating
        # the engagement signal.
        baseline = float(ventral_act.mean())

        TPN = (
            0.40 * (language_act - baseline)
            + 0.40 * (DAN_act    - baseline)
            + 0.20 * (ACC_act    - baseline)
        )
        TNN = DMN_act - baseline

        tpni = (TPN - TNN) / (np.abs(TPN) + np.abs(TNN) + 1e-8)
        return seq_ids_arr, 50.0 * (1.0 + tpni)

    def mean_score(self, scores: np.ndarray) -> float:
        """Returns mean score across sequences. Works with both relative and absolute scores."""
        if len(scores) == 0:
            return 0.0
        return float(scores.mean())
