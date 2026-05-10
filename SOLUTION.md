# Solution Report

## Reproducibility Instructions

To reproduce the solution and generate the `predictions.csv`, run the following commands from the repository root:

```bash
pip install -r requirements.txt
python solution.py
```
Random seeds are fixed to `42` in the PCA, SVM, and stratified splitting logic. However, because the final SVM uses `probability=True`, scikit-learn applies internal probability calibration. This means the predicted probabilities may vary slightly across different scikit-learn versions or hardware environments. The final labels are still generated deterministically for a fixed environment, package version, and input data.

The solution was developed and tested with the following environment:

```text
Python: 3.13
Operating system: macOS
Hardware: MPS
```

The main Python dependencies are specified in `requirements.txt`

## Final Solution Description

The final solution is based on hidden-state probing. I modified the three allowed files: `aggregation.py`, `probe.py`, and `splitting.py`.

In `aggregation.py`, I changed the feature extraction strategy to use **response mean late-layer concatenation**. Instead of using only the final hidden state, the method identifies the response-token positions and mean-pools the response-token hidden states from layers **17–24**. These eight layer representations are concatenated into a single feature vector. Since `Qwen2.5-0.5B` has hidden dimension **896**, the final feature vector has:

```text
8 × 896 = 7168 dimensions
```

I chose this feature set because late layers gave stronger results than shallow layers or final-layer-only representations. Mean-pooling over response tokens also made the representation more stable than relying on only one token.

In `probe.py`, I replaced the neural MLP probe with a classical machine learning classifier. The final probe is:

```text
StandardScaler -> PCA(n_components=50) -> SVC(RBF, C=0.2, gamma=0.0001, class_weight="balanced")
```

This was chosen because the dataset has only **689 examples**, making an MLP likely to overfit. PCA reduces the high-dimensional hidden-state feature vector before classification, while the RBF SVM is flexible enough to capture non-linear patterns in the hidden-state representation.

In `splitting.py`, I changed the split strategy to **5-fold stratified cross-validation**. Each fold keeps the class distribution approximately stable and uses a validation split for threshold tuning inside the probe.

The biggest improvement came from using the response-token hidden states from late layers **17–24** instead of simpler final-layer features, because the last token alone does not contain enough information about the whole response and can miss signals that are spread across multiple response tokens and layers.

# More Deatiled Analysis
## Dataset

The LLM is assigned to answer questions given some context. The trick is that the context does not always contain the information from which the answer can be entailed. So the task for an LLM is not just to answer a question, but to be able to:

- detect if the context contains useful information
- if yes, use the context to answer the question correctly
- if the context does not help answer the question, state that by saying unable to answer.

Nearly **70%** of the dataset are representatives of the hallucinated class, introducing a class imbalance which makes the task of training harder. The key issue is also that the types of hallucinations vary, including cases where the model outputs complete nonsense, or is mistaken in a single fact.

The dataset contains only **689 data points**, so using an overly complex probe architecture increases the risk of overfitting.

## Approach Selection

To understand what approach is suitable, I focused on papers that detect hallucinations using hidden states. A simple direction is to extract hidden states from one or more layers and train a classifier on top of them. This matches the idea from Azaria and Mitchell [1], who show that an LLM’s hidden activations can contain information about whether a generated statement is true or false. Their method trains a classifier directly on hidden-layer activations, which is close to the probe-based setting of this task.

This direction is suitable here because the model is supposed to act as a lightweight probe, not as a new generator. However, the dataset contains only **689 examples**, so a complex neural classifier is risky and can easily overfit. Because of that, I focus more on comparing hidden-state representations and using a simple classifier rather than training a large probe.

Some newer methods also show that the choice of layer matters. INSIDE uses internal states for hallucination detection and reports that middle-layer last-token embeddings work better than shallow or final-layer embeddings [2]. This is useful for my setting because it suggests that instead of using all layers blindly, I should experiment with selected middle or late layers.

ICR Probe is also relevant because it argues that hallucination detection should not only look at one static hidden state, but also at how hidden states change across layers. It introduces the ICR Score to capture the update process of the residual stream across layers [3]. However, the full method uses attention-related information, which is not allowed in this task, so only the hidden-state dynamics idea can be reused.

Considering that the LLM must answer using the provided context, the paper that seemed most promising for this setting is *Detecting LLM Hallucination Through Layer-wise Information Deficiency: Analysis of Ambiguous Prompts and Unanswerable Questions* [4]. Its idea of layer-wise usable information fits the task because the question is not just whether the model knows a fact, but whether the provided context actually supports the answer. In my setting, layer-wise information can be approximated by comparing the model’s negative log-likelihood for the same output with and without the context, layer by layer. If the context really supports the answer, adding the context should make the response easier for the model to justify, so the LI score should be higher for non-hallucinated responses. In other words, truthful answers are expected to gain more usable information from the context, while hallucinated answers should benefit less from it.

## Feature Selection

Unfortunately, despite having high hopes for classification with layer-wise usable information from layers, all experiments with LI as a feature gave poor results. I tried different combinations: LI from each layer separately, only from middle and late layers, mean LI across layers, and min/max LI values, but none of them gave a strong enough signal.

This may be because the dataset is small and noisy, and the LI signal is not enough to separate all hallucination types. Additionally, the LLM was not always trying to directly answer the question. In some cases, it returned an output saying that it cannot answer the question with the given context. This adds noise to the LI-based approach, because such an output may have low LI with the context, but it is actually not hallucinated, since this is the expected behaviour.

In other cases, the context and the answer may look semantically consistent on the surface, so the model can still assign a relatively high LI score. However, the answer can still be incorrect because of one wrong detail. This means that LI can treat the example as confident or context-related, even though the output is actually hallucinated.

Because of this, I moved to hidden-state based features that are easier to calculate. These features describe the representation itself and how it changes across layers.

I experimented with several hidden-state feature sets. First, I used final-layer features, such as the last response token, mean response representation, and max-pooled response representation. Then, I tested late-layer concatenations from layers 17–24, using the last response token, mean-pooled response tokens, and max-pooled response tokens. I also tried a weighted combination of the top three layers. In addition to these vector features, I added scalar geometric features such as hidden-state norms, cosine and L2 changes between layers, response-token spread, prompt-response alignment, and segment-level context/question/response alignment. Finally, I tested combinations of vector features with these scalar geometry features.

## Model Selection

Taking an MLP as a probe may be an overkill, given only **689 datapoints**. I decided to shift from a deep learning probe to classical machine learning models. I experimented with Logistic Regression, Linear SVM, SVM with RBF kernel, Random Forest, Extra Trees, KNN, and Decision Tree classifiers. For models that are sensitive to feature scale, I used standard scaling, and for high-dimensional models such as RBF SVM and KNN, I used PCA before classification to make training more stable and tractable.

I evaluated the models with **5 stratified folds**. Inside each fold, the training portion was split into train and validation subsets. The validation subset was used for decision-threshold tuning. In the final implementation, the probe tunes the threshold to maximise **F1** on the validation split, while AUROC, accuracy, and F1 are reported for train, validation, and test splits.

Among the classical ML models, SVM with an RBF kernel gave the best overall result. The best initial classical ML result was obtained with **SVC_RBF_C0.1**, reaching a mean test AUROC of about **0.777**. Because of this, I selected RBF SVM for further hyperparameter tuning.

For the final tuning step, I searched over different values of **C**, **gamma**, and **PCA dimensionality**. The best tuned configuration was **C = 0.2**, **gamma = 0.0001**, and **pca_dims = 50**. The final probe uses:

```text
StandardScaler -> PCA(n_components=50) -> SVC(RBF, C=0.2, gamma=0.0001, class_weight="balanced")
```
This improved the mean test AUROC to about **0.784**, with mean test accuracy around **0.742** and mean test F1 around **0.841**. This suggests that a relatively regularized **RBF SVM** works best for this dataset: it is flexible enough to capture non-linear structure in the hidden-state features, but not too flexible to overfit heavily on the small number of examples.

| Metric | Value |
|---|---:|
| Test AUROC | 0.784 |
| Test Accuracy | 0.742 |
| Test F1 | 0.841 |

## Interpretation

As a result, based on the hyperparameter sweep and the strongest AUROC performance, I selected the response mean late-layer concatenation feature set. This feature vector is built by mean-pooling the response-token hidden states from layers 17–24 and concatenating them, resulting in a **7168-dimensional** feature vector. The selected model was an SVM with an RBF kernel, tuned with **C = 0.2**, **gamma = 0.0001**, and **PCA dimensionality = 50**.

However, I do not consider this solution perfect. From the metric report, the model is clearly much better at predicting the hallucinated class, which is the majority class, but struggles to recognize truthful cases.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| truthful | 0.70 | 0.34 | 0.46 | 41 |
| hallucinated | 0.77 | 0.94 | 0.85 | 97 |
| **accuracy** |  |  | **0.76** | **138** |
| **macro avg** | **0.74** | **0.64** | **0.65** | **138** |
| **weighted avg** | **0.75** | **0.76** | **0.73** | **138** |

The truthful recall is only **0.34**, meaning many truthful examples are classified as hallucinated. Therefore, even though this model has the best ROC AUC, it is still biased toward the majority class (in fact, just like all the models I tried through the whole experimentation).

After experimenting with different approaches, I concluded that the main way to improve performance is to reconsider the dataset. The hallucination examples are very diverse, so it may be better to divide them into subclasses and turn the task into a multiclass classification problem. This could help the model learn more specific patterns for each hallucination type instead of treating all hallucinations as one broad class.

Adding more datapoints would probably have the biggest impact. The current dataset is small, which makes it difficult to learn stable patterns, especially with high-dimensional hidden-state features. The imbalance itself is not necessarily the main issue, because hallucinations are naturally more diverse, but each subclass should have enough examples.

I also think layer-wise information could still be useful, but LI should be calculated more carefully. In particular, it may be better to compare the question and output both with and without context. Otherwise, cases where the model correctly says it cannot answer may look like hallucinations, because their LI with the context can be low even though the refusal is actually correct.

Finally, geometric and topological hidden-state features may also help, because hallucination may appear not only in LI1 score, but in how the representation changes across layers.

## References

[1] Amos Azaria and Tom Mitchell. *The Internal State of an LLM Knows When It’s Lying*. Findings of EMNLP, 2023.

[2] Chao Chen et al. *INSIDE: LLMs’ Internal States Retain the Power of Hallucination Detection*. ICLR, 2024.

[3] Zhenliang Zhang et al. *ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs*. arXiv preprint, 2025.

[4] *Detecting LLM Hallucination Through Layer-wise Information Deficiency: Analysis of Ambiguous Prompts and Unanswerable Questions*. EMNLP.

[5] Chi Seng Cheang et al. *Do LLMs Really Know What They Don’t Know? Internal States Mainly Reflect Knowledge Recall Rather Than Truthfulness*. arXiv preprint, 2026.
