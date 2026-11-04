"""
Fixes a real bug in YAIB's icu_benchmarks.models.wrappers.MLWrapper.fit():

    if "class_weight" in self.model.get_params().keys():  # Set class weights
        self.model.set_params(class_weight=self.weight)

`self.weight` is a class-level attribute defined on BaseModule as
`weight = None`, and is NEVER reassigned anywhere in the codebase --
confirmed by grepping the entire icu_benchmarks package. The method that
DOES compute real class-balance weights, `set_weight()`, stores its result
in a DIFFERENT attribute, `self.loss_weights` (intended for the
deep-learning loss-weighting path), not `self.weight`. The two names look
similar enough that this reads, at a glance, like class weighting is being
applied -- it never is, for any scikit-learn model in this pipeline
(LogisticRegression, RFClassifier, SVMClassifier, etc.).

CONCRETE IMPACT (confirmed against a real completed run on full MIMIC-III):
mortality24's positive rate is ~8.1% (2,173 of 26,835 training stays).
Without class_weight='balanced' reaching LogisticRegression, the model
optimizes plain unweighted log-loss, which a linear decision boundary can
trivially "solve" by leaning toward the majority class -- producing the
near-chance AUROC (0.49-0.55) with near-zero variance observed across all
three tasks (mortality, sepsis, AKI) for LogisticRegression specifically,
while LGBMClassifier and LSTM (both more robust to this failure mode)
landed in the expected 0.81-0.89 range on the same data and split.

WHY THE FIX PASSES THE STRING "balanced" RATHER THAN A COMPUTED WEIGHT
DICT: get_balance() (icu_benchmarks/data/loader.py) returns a plain list
of two floats via manual inverse-frequency calculation -- not the
{0: w0, 1: w1} dict format scikit-learn's class_weight parameter actually
expects, and scikit-learn models don't accept a bare list here at all.
Rather than reproduce that list->dict conversion (and risk a second,
subtler formatting bug), this fix passes literal "balanced" through
directly -- scikit-learn's own native, well-tested balancing computes the
equivalent weighting internally from whatever data is passed to .fit(),
sidestepping YAIB's broken custom weight-threading path entirely for
scikit-learn-backed models.

WHY A MONKEYPATCH RATHER THAN EDITING external/YAIB DIRECTLY: same
reasoning as apply_fixed_split.py -- external/YAIB is a tracked
submodule/checkout; a monkeypatch applied from this project's own code,
imported before training runs, keeps the fix version-controlled here
rather than as an easy-to-lose local edit to the upstream checkout.

USAGE: import and call patch() before any MLWrapper-derived model
(LogisticRegression, RFClassifier, etc.) is instantiated or trained --
concretely, alongside apply_fixed_split.patch() in the same python -c
preamble used for training jobs, since both need to be in effect before
icu_benchmarks.run.main() runs.

    import apply_fixed_split
    apply_fixed_split.patch(split_path=...)
    import fix_class_weight_bug
    fix_class_weight_bug.patch()
    from icu_benchmarks.run import main
    main(my_args=[...])

CAVEAT: this patches the ml_models.wrappers.MLWrapper.fit method (the base
class covering LogisticRegression, RFClassifier, SVMClassifier, and any
other scikit-learn-backed model in this pipeline), not just
LogisticRegression specifically -- if RFClassifier or another
scikit-learn model is used elsewhere in this project and its own results
were NOT showing the same near-chance symptom, that's worth a second look
before assuming this patch changes nothing for it: tree ensembles are
generally more robust to imbalance even without explicit weighting, so an
unaffected-looking RFClassifier run under the OLD (buggy) code doesn't
necessarily mean this fix is a no-op for it -- it may simply have been
less exposed to the same failure mode, and should still benefit from
correct weighting once this patch is applied.
"""

import logging

logger = logging.getLogger(__name__)


def patch():
    """
    Applies the monkeypatch. Call this ONCE, before any MLWrapper-derived
    model is trained, so the corrected fit() is what actually executes.
    """
    from icu_benchmarks.models.wrappers import MLWrapper

    original_fit = MLWrapper.fit

    def _patched_fit(self, train_dataset, val_dataset):
        """
        Same as the original MLWrapper.fit(), except class_weight is set
        to "balanced" directly (scikit-learn's native mechanism) instead
        of the always-None self.weight attribute the original code reads
        from. Everything else is unchanged from upstream.
        """
        train_rep, train_label, row_indicators = train_dataset.get_data_and_labels()
        val_rep, val_label, row_indicators = val_dataset.get_data_and_labels()

        self.set_metrics(train_label)

        if "class_weight" in self.model.get_params().keys():
            self.model.set_params(class_weight="balanced")
            logger.info(
                f"[fix_class_weight_bug] Set class_weight='balanced' on "
                f"{type(self.model).__name__} (upstream YAIB would have set "
                f"class_weight=None here due to the self.weight/self.loss_weights "
                f"attribute-name bug -- see this module's docstring)."
            )

        val_loss = self.fit_model(train_rep, train_label, val_rep, val_label)

        train_pred = self.predict(train_rep)

        logging.debug(f"Model:{self.model}")

        self.log("train/loss", self.loss(train_label, train_pred), sync_dist=True)
        logging.debug(f"Train loss: {self.loss(train_label, train_pred)}")
        self.log("val/loss", val_loss, sync_dist=True)
        logging.debug(f"Val loss: {val_loss}")
        self.log_metrics(train_label, train_pred, "train")

    MLWrapper.fit = _patched_fit
    logger.info(
        "[fix_class_weight_bug] Patched MLWrapper.fit to use class_weight='balanced' "
        "for scikit-learn-backed models (fixes a real upstream bug -- see module docstring)."
    )
