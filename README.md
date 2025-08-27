# Automated Essay Scoring Engine 

This project is an end-to-end ML application for automatic essay scoring based on the IELTS academic grading rubric.

Our model achieved a **QWK of 0.49** and **100% accuracy** within a 1.0 score band on the held-out test set.

---

## Live Demo
<img src="./assets/demo.gif" width="6000">

---

## How to Run 
To run this project locally, please follow these steps. The recommended method is using Conda and the `environment.yml` file, as it guarantees correct installation.


**Clone the repo**
```bash
git clone https://github.com/mordvi9/Automated_Essay_Grading
cd Automated_Essay_Grading
```
**Create the Conda env**
```bash
conda env create -f environment.yml
```
**Activate**
```bash
conda activate aes_project
```
**Run the app**
```bash
streamlit run src/app.py
```

---


## Results: 
The final model (an XGBoost regressor on our full feature set) significantly outperforms standard lexical (TF-IDF) and modern semantic (MiniLM Transformers) baselines. Our model's 0.49 QWK improves 227% on the 0.15 QWK achieved by the baselines. This proves the value of task-specific feature engineering for this problem.

<img src="./assets/baseline_comp.png" alt="Comparison with Baselines" width="450">

---

## Method & Analysis

**1. Data**

We extracted numerical scores from detailed feedback on the scored IELTS academic corpus. We used 5000 essays along with their prompts and overall score bands to train our model. 

**2. Features**

We analyzed the importance of different feature categories. The study confirmed that our complete feature set ("All Features") was essential for achieving the best performance.

Our Features include:

**Linguistic:** Syntactic complexity, POS tag ratios, readability scores (Flesch).

**Lexical:** TF-IDF, word repetition, and lexical diversity metrics.

**Semantic:** BERT embeddings for prompt-essay similarity.

**Grammatical:** Custom rule-based error detection.


<img src="./assets/features.png" alt="Feature ablation study" width="300">

**3. Model Selection**

We compared several models on the task. The learning curves clearly show that XGBoost provided the most robust and highest-performing solution for this feature set.

<img src="./assets/model_comp.png" alt="Model learning curves" width="300">

**4. Final Model Precision**

The final model is highly precise, with 100% of predictions falling within the ±1.0 score band (see baseline prediction comparison plots).


---
