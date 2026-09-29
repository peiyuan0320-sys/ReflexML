<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# ReflexML LR-Switch v1 secondary-outcome sign clarification

- Protocol ID: `ReflexML-Mechanism-LR-Switch-v1`
- Frozen protocol SHA-256: `2c5d58f45c5db454eb091023201a40d49cf0b75fd783c9a826a596fe8c217f48`
- Date and context: 2026-09-26, before opening or aggregating the real production secondary outcomes; clarification of the already-prespecified Section 9 analysis rule.

Section 9 of the frozen protocol says exactly:

> Pre-specified secondary outcomes are:
>
> 1. epoch18 training loss;
> 2. epoch18 validation accuracy.
>
> For each outcome, use the same paired Switch-vs-Continue construction.

Section 7 defines that construction for the primary validation-loss contrast as `Continue - Switch`. For both secondary outcomes, retain the same arithmetic treatment ordering:

- epoch18 training loss: `train_loss_continue - train_loss_switch`;
- epoch18 validation accuracy: `val_accuracy_continue - val_accuracy_switch`.

Thus a positive training-loss contrast means lower training loss under Switch, while a negative validation-accuracy contrast means higher accuracy under Switch. This is a literal, outcome-independent preservation of the already-defined Switch-vs-Continue construction; the accuracy sign is not reversed to favor a result.

As stated in the pre-unblinding clarification request, no real production secondary aggregate had been calculated or interpreted before this clarification. This record does not alter the frozen protocol, primary estimand, frozen primary result, or its interpretation.
