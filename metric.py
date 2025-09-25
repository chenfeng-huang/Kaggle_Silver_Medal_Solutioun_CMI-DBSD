"""
https://www.kaggle.com/code/metric/cmi-2025

Hierarchical macro-averaged F1 implementation for the CMI 2025 challenge.

Core logic:
1. Compute binary F1 (target vs non-target)
2. Compute multiclass macro F1 (map all non-target to a single 'non_target' class)
3. Final score is the equal-weight average of the two.
""" 

import pandas as pd
from sklearn.metrics import f1_score


class ParticipantVisibleError(Exception):
    """Errors raised here will be shown directly to the competitor."""
    pass


class CompetitionMetric:
    def __init__(self):
        self.target_gestures = [  # Target gesture classes: used as positive/fine classes in binary/multiclass
            'Above ear - pull hair',  # Target: above ear — pull hair
            'Cheek - pinch skin',  # Target: cheek — pinch skin
            'Eyebrow - pull hair',  # Target: eyebrow — pull hair
            'Eyelash - pull hair',  # Target: eyelash — pull hair
            'Forehead - pull hairline',  # Target: forehead — pull hairline
            'Forehead - scratch',  # Target: forehead — scratch
            'Neck - pinch skin',  # Target: neck — pinch skin
            'Neck - scratch',  # Target: neck — scratch
        ]
        self.non_target_gestures = [  # Non-target gesture classes: mapped to 'non_target' in multiclass
            'Write name on leg',  # Non-target: write name on leg
            'Wave hello',  # Non-target: wave hello
            'Glasses on/off',  # Non-target: glasses on/off
            'Text on phone',  # Non-target: text on phone
            'Write name in air',  # Non-target: write name in air
            'Feel around in tray and pull out an object',  # Non-target: feel in tray and pull out object
            'Scratch knee/leg skin',  # Non-target: scratch knee/leg skin
            'Pull air toward your face',  # Non-target: pull air toward your face
            'Drink from bottle/cup',  # Non-target: drink from bottle/cup
            'Pinch knee/leg skin'  # Non-target: pinch knee/leg skin
        ]
        self.all_classes = self.target_gestures + self.non_target_gestures

    def calculate_hierarchical_f1(
        self,
        sol: pd.DataFrame,
        sub: pd.DataFrame
    ) -> float:

        # Validate that predicted gestures fall within the defined set
        invalid_types = {i for i in sub['gesture'].unique() if i not in self.all_classes}  # Unique predictions not in known classes
        if invalid_types:  # If any invalid classes exist
            raise ParticipantVisibleError(
                f"Invalid gesture values in submission: {invalid_types}"
            )

        # Binary F1 (target vs non-target): map each sample to boolean label
        y_true_bin = sol['gesture'].isin(self.target_gestures).values
        y_pred_bin = sub['gesture'].isin(self.target_gestures).values
        # Compute binary F1
        f1_binary = f1_score(
            y_true_bin,
            y_pred_bin,
            pos_label=True,
            zero_division=0,
            average='binary'
        )

        # Multiclass labels: keep target classes, merge non-target into 'non_target'
        y_true_mc = sol['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        y_pred_mc = sub['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        # Macro F1: F1 per class averaged equally
        f1_macro = f1_score(
            y_true_mc,
            y_pred_mc,
            average='macro',
            zero_division=0
        )
        
        # Final score: 50% binary F1 + 50% macro F1
        return 0.5 * f1_binary + 0.5 * f1_macro
