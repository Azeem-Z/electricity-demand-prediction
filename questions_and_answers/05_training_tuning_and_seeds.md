# 5. Training, tuning and seeds

## Training

#### Q: What does "training a model" mean?
The model looks at many examples (input -> correct answer) from the training part and changes its inside numbers
(tree splits or neural network weights) so its answers get closer to the correct ones.
Example: input = last 72 hours of PJM demand, correct answer = demand 24 hours later.

#### Q: How were the tree models trained?
- Random Forest: `RandomForestRegressor(...).fit(X_train, y_train)` with the tuned settings, each tree on half the rows.
- XGBoost: `XGBRegressor(...).fit(X_train, y_train)` with the tuned settings and the tuned number of trees.
- One model per dataset, per horizon, per seed (2 x 3 x 3 = 18 models for each of the two).

#### Q: How were the neural networks trained?
| Setting | Value | Meaning |
|---|---|---|
| Loss | MSE (mean squared error) | the number the network tries to make small |
| Optimizer | Adam | the method that changes the weights after each batch |
| Learning rate | tuned (0.0003 to 0.007) | how big each change of the weights is |
| Batch size | 512 | number of examples looked at before each weight change |
| Epochs | at most 10 (4 during tuning) | one epoch = the model has seen every training example once |
| Early stopping | patience 2, keep best weights | stop if validation loss does not improve for 2 epochs |
| Shuffle | yes, with the seed | training windows come in random order each epoch |

#### Q: What is MSE loss?
Mean Squared Error = average of (prediction - actual)^2. Squaring makes big mistakes count much more.
Example: errors 1 and 3 -> (1 + 9) / 2 = 5.

#### Q: What is Adam?
Adaptive Moment Estimation. The most common optimizer. It changes each weight with its own step size, based on how the
errors behaved in recent steps. It usually trains faster and more stably than plain gradient descent.

#### Q: What is an epoch and a batch? Give an example.
PJM 24h has about 35,000 training windows. With batch 512, one epoch = about 69 weight updates (35,000 / 512).
10 epochs = about 690 updates.

#### Q: What is early stopping?
After each epoch we check the loss on the validation data. If it does not get better for 2 epochs in a row, training
stops and the weights from the best epoch are kept (`restore_best_weights=True`). This stops overfitting.

#### Q: What is overfitting?
The model learns the training data by heart, including its noise, and then does badly on new data.
Signs: training error keeps going down but validation error goes up.

#### Q: How is a window made for the neural networks?
For a target hour t and horizon h, the window is the L hours that end h hours before t.
Example: L = 48, h = 24, target = Friday 18:00 -> window = Tuesday 19:00 to Thursday 18:00.
In the code the windows are cut on the fly from one long table (`make_ds` with `tf.gather`), so we do not need to
store millions of copies.

## Tuning

#### Q: What is hyperparameter tuning?
Hyperparameters are settings we choose before training (number of trees, learning rate, window length...).
The model does not learn them itself. Tuning = trying different settings and keeping the one with the lowest
validation error.

#### Q: Which tool did you use for tuning?
Optuna with the TPE sampler (Tree-structured Parzen Estimator). This is a Bayesian method: after each try it looks at
which settings gave good results and tries settings close to those more often. It finds good settings faster than
random search or grid search.

#### Q: How many tries? Why so few?
8 trials per model per dataset, for every model. The budget is small but the same for all models, so the comparison is
fair: no model got more tuning than another. This is listed as a limitation.

#### Q: On which horizon did you tune?
Only at 24 hours, then the same settings were used for 1 and 168 hours. This keeps the budget small. Tuning used seed 0
and the validation data only, never the test data.

#### Q: What did you search over?
| Model | Settings searched |
|---|---|
| Random Forest | trees 100-400, max depth 5-30, min samples per leaf 1-20, max features 0.3/0.5/0.8/1.0 |
| XGBoost | learning rate 0.01-0.3 (log), max depth 3-10, min child weight 1-20, subsample 0.5-1, colsample 0.5-1, L2 penalty 0.001-10 (log); number of trees by early stopping (up to 2000, stop after 50 rounds without improvement) |
| All neural | window L 24/48/72, learning rate 0.0001-0.01 (log), dropout 0-0.3, units 32/64/128 |
| LSTM / GRU | + layers 1 or 2 |
| CNN-LSTM | + filters 32/64, kernel 3/5 |
| Attention | + heads 2/4 |

#### Q: What does "log" in the search mean?
The search picks values evenly on a log scale. So between 0.0001 and 0.01 it tries 0.0001-0.001 as often as 0.001-0.01.
On a normal scale almost all tries would be above 0.001 and the small values would never be tested.

#### Q: Did you use any logging?
Only to keep the output short: `optuna.logging.set_verbosity(WARNING)` and the TensorFlow logger set to "ERROR", so the
notebooks do not print hundreds of lines for every trial. Training progress was printed as one line per model
(validation RMSE and time). `verbose=0` in Keras hides the epoch progress bars. There were no log files.

## Seeds

#### Q: What is a random seed?
A starting number for the random number generator. Same seed = same "random" numbers = same result every time.
Example: `random_state=0` in Random Forest always picks the same random rows.

#### Q: Where is randomness in your models?
- Random Forest: which rows and columns each tree gets.
- XGBoost: `subsample` and `colsample_bytree` pick random rows and columns.
- Neural networks: starting weights, dropout, order of the training windows.
- Seasonal naive: no randomness (so it has only one run).

#### Q: Why three seeds (0, 1, 2)?
One run can be lucky or unlucky. We train every model 3 times with seeds 0, 1 and 2 and report the mean.
This also shows how stable a model is.
Example, PJM 1 hour ahead RMSE:
| Model | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Random Forest | 527 | 532 | 536 |
| Attention | 1219 | 1248 | 962 |
The trees hardly change, the attention model changes a lot. Neural networks are less stable with a short training budget.

#### Q: Why seed 42 in Optuna?
`TPESampler(seed=42)` makes the tuning repeatable. 42 is just a common choice, any fixed number works.

#### Q: How did you set the seed for the neural networks?
`keras.utils.set_random_seed(seed)` sets the seed for Python, NumPy and TensorFlow together, and the training data
shuffle uses the same seed.
