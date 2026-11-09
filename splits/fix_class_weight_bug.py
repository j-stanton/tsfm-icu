"""Patch YAIB's MLWrapper.fit so class weights reach scikit-learn models."""

import logging

logger = logging.getLogger(__name__)


def patch():
    """Applies the monkeypatch."""
    from icu_benchmarks.models.wrappers import MLWrapper

    original_fit = MLWrapper.fit

    def _patched_fit(self, train_dataset, val_dataset):
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
