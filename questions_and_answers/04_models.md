# 4. Models (notebooks 04 and 05)

We compare 7 models in 3 groups:
| Group | Models | Input |
|---|---|---|
| Benchmark | seasonal naive | demand one week earlier |
| Tree models | Random Forest, XGBoost | lag columns + rolling means + calendar |
| Neural networks | LSTM, GRU, CNN-LSTM, attention (simplified TFT) | window of the last 24-72 hours + target-hour calendar |

## Benchmark

#### Q: What is the seasonal naive model?
The simplest possible forecast: "demand will be the same as at the same hour one week ago" (lag_168).
Example: to predict Monday 18:00, take last Monday 18:00.
It has no training. Every real model must beat it, otherwise the model is useless.
It is the same value for all three horizons, so its score is the same at 1, 24 and 168 hours.

## Tree models

#### Q: What is a decision tree?
A list of yes/no questions that ends in a number.
Example: "Is lag_24 > 35,000 MW?" yes -> "Is hour > 15?" yes -> predict 41,000 MW.

#### Q: What is Random Forest?
Many decision trees (100 for PJM, 350 for UCI after tuning). Each tree is trained on a random half of the rows
(`max_samples=0.5`) and looks at a random part of the columns at each split (`max_features`). The final forecast is the
average of all trees. Many slightly different trees together make fewer mistakes than one tree.

#### Q: What is XGBoost?
XGBoost = Extreme Gradient Boosting. Trees are built one after another. Each new tree tries to fix the errors that the
trees before it still make. The `learning_rate` (about 0.01 here) says how much each new tree is allowed to change the answer.
Small steps, many trees (326 for PJM, 328 for UCI).

#### Q: Difference between Random Forest and XGBoost in one line?
Random Forest: trees built at the same time and averaged. XGBoost: trees built in a row, each one correcting the last.

#### Q: What is feature importance?
How much each input column helped the tree model. Example: for PJM 24 hours ahead, lag_24 (same hour yesterday) is most
important. For UCI the hour of day and the demand one and two weeks earlier are most important.

## Neural networks

#### Q: What is a neural network?
A model made of layers of small calculators (neurons). Each neuron multiplies its inputs by weights, adds them up and
passes the result on. During training the weights are changed little by little so the prediction gets closer to the
real value.

#### Q: What is an RNN?
Recurrent Neural Network. It reads the hours one by one in time order and keeps a "memory" (hidden state) of what it
has seen so far. Good for sequences like demand over time.

#### Q: What is LSTM?
Long Short-Term Memory. An RNN with three gates that decide what to forget, what to store and what to output. This helps
it remember patterns from many hours back (e.g. the same hour yesterday), which a plain RNN forgets.

#### Q: What is GRU?
Gated Recurrent Unit. A simpler LSTM with two gates instead of three. Fewer weights, so it trains faster and often works
just as well. In our results the GRU was the best neural model.

#### Q: What is CNN-LSTM?
A 1-D convolution layer first finds short shapes in the window (e.g. a sharp morning rise), then an LSTM reads these
shapes in order. CNN = Convolutional Neural Network.

#### Q: What is the attention model (simplified TFT)?
A model based on the Temporal Fusion Transformer. An LSTM reads the window, then an attention layer gives a weight to
every past hour, saying how important it is for the forecast. "Simplified" because the full TFT has more parts
(variable selection networks, static covariates, quantile outputs) that we left out to keep it small and fair.

#### Q: What is the same in all four neural models?
- Same input window (last L hours, 8 values per hour).
- Same output part: join with the 7 calendar values of the target hour -> Dense layer (32 units, ReLU) -> Dense layer (1 unit) -> forecast.
- Same loss (MSE), optimizer (Adam), batch size (512), tuning budget, early stopping and seeds.
Only the middle layers differ. This is shown in Fig. 2 of the paper.

## Layer names explained (from the model picture)

#### Q: What is a dense layer?
The normal neural network layer. Every input is connected to every neuron.
"32 units" = 32 neurons, so it outputs 32 numbers. "1 unit" = 1 neuron, so it outputs one number: the forecast.

#### Q: What is ReLU?
Rectified Linear Unit. A simple rule applied to each number: negative -> 0, positive -> stays the same.
Example: -3 -> 0, 5 -> 5. Without such a rule the whole network could only learn straight-line relationships.
(The attention model also uses ELU in its feed-forward block, a smooth version of ReLU.)

#### Q: What is dropout?
During training, a random part of the neurons is switched off at each step (e.g. dropout 0.05 = 5%). The network cannot
rely on one neuron, so it learns more general patterns and overfits less. Dropout is off when predicting.

#### Q: What is 1-D convolution?
A small window (the kernel, 5 hours here) slides along the hours and multiplies them by learned weights, like a pattern
detector. 64 filters = 64 different pattern detectors. "1-D" = it slides in one direction (time). Images use 2-D.
"Causal" padding means it only looks at past hours, never future ones.

#### Q: What is max pooling?
Takes the largest value of every 2 neighbouring hours (`MaxPooling1D(2)`). The sequence gets half as long, keeps the
strongest signals and makes the LSTM after it faster. Example: [3, 7, 2, 5] -> [7, 5].

#### Q: What is dense embedding?
Each hour has 8 input numbers. A dense layer changes them into a bigger set of numbers (e.g. 128) for that hour, which
the next layers can work with more easily. "Embedding" = this new representation of each hour.

#### Q: What is "gate, add and norm"?
Three small steps after a layer:
- Gate: a learned switch between 0 and 1 (sigmoid) that decides how much of the new output to keep. If a layer is not
  useful, the gate can close it.
- Add: the input of the layer is added back to its output (skip connection), so old information is not lost.
- Norm: layer normalization, rescales the numbers to a similar range so training stays stable.

#### Q: What is multi-head attention?
Attention gives each past hour a weight (all weights add up to 1) and makes a weighted average. Hours that matter more
get bigger weights. "Multi-head" = several attentions in parallel (2 heads here), each can look for a different pattern.
In our model the last hour of the window asks "which past hours are useful for me?" (query) and all hours answer (keys).

#### Q: What is the feed-forward block?
Two dense layers after the attention (ELU, then linear). It processes the result of the attention once more.

#### Q: What are the tuned settings of the neural models?
From `results/deep_params.json` (L = window length in hours, lr = learning rate):
| Model | PJM | UCI |
|---|---|---|
| LSTM | L=72, 128 units, 1 layer | L=48, 128 units, 2 layers |
| GRU | L=72, 128 units, 1 layer | L=72, 128 units, 2 layers |
| CNN-LSTM | L=48, 64 filters, kernel 5 | L=72, 64 filters, kernel 5 |
| Attention | L=48, 128 units, 2 heads | L=48, 32 units, 2 heads |
