import numpy as np
import time


class TRIBERunner:
    def __init__(self, config):
        """
        TODO: Load TRIBE v2 from config.TRIBE_MODEL_PATH onto config.DEVICE.
        Example:
            from tribev2 import TribeV2Model
            self.model = TribeV2Model.from_pretrained(config.TRIBE_MODEL_PATH)
            self.model.to(config.DEVICE)
            self.model.eval()
        Set self.ready = True when loaded successfully.
        """
        # TODO: replace with real model loading
        self.config = config
        self.model = None
        self.ready = False

    def run(self, preprocessed: dict) -> np.ndarray:
        """
        TODO: Run TRIBE v2 forward pass.
        Stack the preprocessed input N_SUBJECTS times along the batch dimension,
        run a single forward pass (all subjects at once — this is the MI300X workload),
        average activations across the subject dimension.

        Example sketch:
            import torch
            batch = stack_for_subjects(preprocessed, self.config.N_SUBJECTS)
            with torch.no_grad():
                activations = self.model(batch)          # (N_SUBJECTS, n_seg, 70000)
            return activations.mean(dim=0).cpu().numpy() # (n_seg, 70000)

        Returns np.ndarray of shape (n_segments, 70000) — one voxel activation
        vector per text segment, averaged across virtual subjects.
        """
        # TODO: replace with real forward pass
        time.sleep(0.4)  # MOCK: simulate inference latency
        n_segments = len(preprocessed.get("segments", ["placeholder"]))
        return np.random.rand(n_segments, 70000).astype(np.float32)
