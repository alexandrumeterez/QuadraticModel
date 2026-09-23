I want a script that can load a checkpoint and do the following runs:

There are three options for the model+loss transform:
- "none": no transformation, just load the model and run it as is
- "quad": linearize the model using the method in linearization2, and use the quadratic loss
- "prox": linearize the model using the method in linearization2, but keep cross entropy loss

There are two options for which params to train:
- "full": train all params
- "head": only train the head params (last layer)

There are three options for the optimizer:
- "adam": use the Adam optimizer with the hparams from the checkpoint
- "stale_adam": like Adam but it updates (w,nu) together rather than sequentially, so the updates go mu -> (w,nu). Effectively it uses a stale nu to update w.
- "frozen_adam": use the Adam optimizer from the checkpoint but freeze nu. Make sure that mu still evolves
- "sgdm": run stale_adam with nu frozen, which is effectively the same as SGD with momentum and a fixed preconditioner in terms of the frozen nu

Note that the difference between frozen_adam and SGDM is that the frozen Adam update is:
$$
\begin{align*}
m &\gets \beta_1 m + (1-\beta_1) g \\
w &\gets w - \frac{m}{\sqrt{\beta_2 \nu + (1-\beta_2) g^2}}
\end{align*}
$$
and the SGDM update is:
$$
\begin{align*}
m &\gets \beta_1 m + (1-\beta_1) g \\
w &\gets w - \frac{m}{\sqrt{\nu}}
\end{align*}
$$

Gradient clipping should have two CLI options: type and threshold. The type can be "none", "clip", or "skip". The threshold is a positive float that determines the clipping threshold. Mathematically, clipping should be based on the following criterion: for each param group (embd,head,blocks.mlp.up, etc) compute mean(nu) and mean(g^2). There are three options for type:
- "none": no gradient clipping
- "skip": if mean(g^2) > threshold * mean(nu) for **any** layer, then skip the batch (no update)
- "clip": for any layer with mean(g^2) > threshold * mean(nu), scale the gradients so that mean(g^2) = threshold * mean(nu)

# Code Instructions
- The JAX venv is in `$VENVS/jax`. It is based on UV and has all the libraries installed.
- You may or may not have access to a GPU. If you don't, don't run anything intensive, only basic checks. If you do and you think it would be helpful, you can run simple tests (< 1min) to verify the code does what it should
- Keep code minimal. Don't worry about edge cases, assume the person running the script will provide valid inputs and is sane. Don't overly comment the code. Use the current ../pretrain.py and ../transformer.py for inspiration for code style.