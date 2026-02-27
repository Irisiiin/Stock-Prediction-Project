# CS1671 Project Proposal: Predicting Daily S&P 500 Index Direction from Financial News Headlines

## 1. Task: What is the problem or task you are focusing on?

We are focusing on a binary classification task in natural language processing applied to financial prediction: predicting the daily directional movement (up or down/flat) of the S&P 500 index based on financial news headlines from the same trading day. Specifically, we aim to determine whether the S&P 500 closing price will increase (up) or decrease/remain flat (down/flat) on the next trading day, using textual features extracted from news headlines to capture market sentiment. This task belongs to the supervised discriminative classification framework covered in class. It directly addresses the project prompt to "predict stock price change or other market behavior from public datasets of financial news," emphasizing how NLP can model short-term, sentiment-driven market reactions. We keep the task binary to ensure simplicity and evaluability, avoiding regression on magnitude or multi-day horizons.

## 2. Input and output: What is the format of the input and output of this task?

The input consists of financial news headline text from a given trading day. When multiple headlines appear for the same day (as seen in some dataset entries), we concatenate them into a single document string for processing. This aggregated text is then fed into NLP techniques to extract features such as sentiment scores, n-grams, or embeddings.

The output is a binary label from the discrete set {0, 1}, where:
- 1 indicates the S&P 500 closing price increases on the next trading day (`close_{t+1} > close_t`),
- 0 indicates the closing price decreases or stays the same (`close_{t+1} ≤ close_t`).

**Example** (based on real dataset patterns; values approximated from previews):  
- **Input**: "JPMorgan Predicts 2008 Will Be 'Nothing But Net' Dow Tallies Biggest First-Session-of-Year Point Drop Ever" (concatenated headlines from the same day)  
- **Output**: 0 (reflecting that the next trading day's S&P 500 close declined or was flat relative to the prior close).

Another example:  
- **Input**: "Fed Signals Rate Cut Amid Economic Concerns"  
- **Output**: 1 (illustrative case where positive policy news correlates with an upward movement).

## 3. Data: What data are you using?

We are primarily using the **S&P 500 with Financial News Headlines (2008–2024)** dataset, available at https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024. This dataset contains approximately 19,000+ rows (exact count ~19,127), where each row corresponds to one financial news headline paired with the S&P 500 closing price on that trading date.

The dataset has three columns:  
- `date`: the trading date in YYYY-MM-DD format,  
- `headline`: the financial news headline as a text string,  
- `close`: the S&P 500 closing price (numeric, in points).

We derive the target labels ourselves: label = 1 if the next day's closing price exceeds the current day's (`close_{t+1} > close_t`), otherwise 0. The distribution is roughly balanced due to the long-term upward bias of the market combined with daily randomness: approximately 52% up (1) and 48% down/flat (0) across the full dataset. Exact counts will be computed for train and development sets during preprocessing.

A very small subset of the data is shown below:

| Date       | Headline                                                                 | Close    |
|------------|--------------------------------------------------------------------------|----------|
| 2008-01-02 | JPMorgan Predicts 2008 Will Be "Nothing But Net"                        | 1447.16 |
| 2008-01-02 | Dow Tallies Biggest First-Session-of-Year Point Drop Ever               | 1447.16 |
| 2008-01-03 | Fed Signals Rate Cut Amid Economic Concerns                             | 1447.12 |

(Note: Some days have multiple headlines; we will concatenate them during preprocessing.)

The dataset was uploaded to Kaggle by user Dyuti Dasmahaptra and constructed by aggregating historical daily financial news headlines (likely from sources such as Reuters, Bloomberg, or market recap sites) and aligning them with official S&P 500 closing prices sourced from Yahoo Finance or similar APIs. It covers trading days from 2008 to 2024, providing long-term coverage of market events. There are no pre-annotated "gold" labels for direction or sentiment; we create our own deterministic binary direction labels from the `close` column.

## 4. Methods: What approach are you taking to building a NLP system to handle this task?

We will build an NLP pipeline combining a traditional statistical baseline with LLM-enhanced methods (to be fully implemented after LLMs are covered in class).

For feature extraction from text, we will use n-gram representations (unigrams and bigrams via TF-IDF), sentiment polarity and subjectivity scores (VADER or TextBlob for the baseline), and optionally preserve numeric tokens (e.g., percentages, rates) or apply keyword weighting for domain-specific entities (e.g., "Fed", "Rate Cut", "Inflation") to better capture financial sensitivity.

The baseline model will be TF-IDF vectors fed into a Support Vector Machine classifier with a linear kernel (sklearn.svm.LinearSVC). SVM is chosen as a strong traditional text classification approach that often outperforms simpler models like Logistic Regression on high-dimensional sparse TF-IDF features, while remaining fully interpretable and within the scope of statistical methods covered in class. We will use the linear kernel to ensure training efficiency and comparability to other linear classifiers discussed.

For the advanced component, we will apply zero-shot or few-shot prompting with financial-tuned LLMs (e.g., FinBERT or accessible models via Hugging Face) for direct direction prediction or as a source of sentiment features to augment the pipeline.

Software packages include Python with pandas (data handling), scikit-learn (baseline models and TF-IDF), NLTK/TextBlob/VADER (sentiment), Hugging Face Transformers (LLM prompting/pipelines), and matplotlib/seaborn (visualization).

## 5. Evaluation: How are you evaluating your approach? What performance metrics are you going to use?

We will evaluate using standard binary classification metrics on a chronological train/dev/test split (e.g., 2008–2018 train, 2019–2021 dev, 2022–2024 test) to prevent lookahead bias. Primary metrics are Accuracy and macro-averaged F1-score (to account for slight imbalance). Additional metrics include Precision, Recall, Confusion Matrix, and ROC-AUC. We will compare results against random guessing (50%) and majority-class baselines. Precision will be particularly emphasized, as in financial prediction contexts, correctly identifying true positives (predicting "up" when the index actually rises) is often more critical than overall accuracy to avoid costly false positives. Results will be reported in tables and graphs, including ablation studies on feature contributions (e.g., sentiment-only vs. full text).

## 6. Ethics: What kinds of ethical issues may be raised by your model or data?

Several ethical concerns may arise. News headlines can reflect sensationalism, political slant, or corporate influence, potentially causing the model to amplify biased market signals. If deployed, the model could be misused to generate false signals or contribute to market manipulation. LLM-based components risk hallucinations that produce misleading financial interpretations. There is also a broader societal risk of reinforcing inequality if such models disproportionately benefit certain market participants. We mitigate these issues by relying exclusively on public datasets, clearly discussing limitations in our report, and explicitly avoiding any claims about real-time trading applicability.

## 7. Steps: What are the proposed steps needed for completion of the project?

1. Download and load the dataset using pandas read_csv.  
2. Clean the data: handle missing values, standardize dates, and remove text artifacts (e.g., quotes).  
3. Engineer binary direction labels by shifting closing prices and applying the comparison rule.  
4. Preprocess text: clean headlines and concatenate multiple headlines per day into a single document.  
5. Perform exploratory data analysis: compute label distribution, generate word clouds, and report basic statistics.  
6. Extract features: compute TF-IDF/n-grams, sentiment scores, and optional keyword/numeric enhancements.  
7. Build and train the baseline SVM model with hyperparameter tuning.  
8. Implement and test LLM prompting (zero-shot/few-shot templates) and run inference.  
9. Evaluate models on the test set, compute metrics, and create confusion matrices/plots.  
10. Compare baseline vs. LLM performance, conduct ablation studies, and perform robustness checks.  
11. Write report sections and prepare presentation slides.

## 8. Roles: What are roles and tasks of each person in the group?

- **Irisin**: Modeling & Experiments Lead + Presentation Lead  
  Designs the overall experimental pipeline, implements and trains models (baseline SVM + LLM), runs evaluations and comparisons, and leads all presentations (slides + Q&A).

- **Yifei**: Data Lead  
  Acquires, cleans, and aligns datasets, engineers labels, performs EDA and statistics, and writes the Data section.

- **Asta**: Baseline & Feature Engineer  
  Implements n-gram/TF-IDF features and sentiment extraction (sklearn/VADER/TextBlob), builds and runs the baseline SVM classifier, and documents baseline methods and results.

- **Ciara**: LLM Prompting Specialist + Ethics Lead  
  Designs and tests LLM prompts (zero/few-shot/CoT), runs inference, evaluates outputs, and writes the Ethics, Limitations, Bias, and Future Work sections.

- **Sanjana**: Report & Visualization Coordinator  
  Manages the ACL LaTeX template, creates figures/tables (matplotlib/seaborn), handles appendix and formatting, and contributes to/polishes the Introduction, Motivation, Related Work, and Conclusion sections.