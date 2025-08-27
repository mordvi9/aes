import pandas as pd
import spacy
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from collections import Counter
import numpy as np
import language_tool_python
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path 
import sys


ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / 'data'
sys.path.append(str(ROOT_DIR / 'src'))
# -------------------------

class TransformerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, X=None, y=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        return self

    def transform(self, texts):
        enc = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=512, return_tensors='pt'
        )
        ids  = enc['input_ids'].to(self.device)
        mask = enc['attention_mask'].to(self.device)
        with torch.no_grad():
            out = self.model(ids, attention_mask=mask)
        return out.last_hidden_state[:,0,:].cpu().numpy()
    
num_rows = 5000

tfidf_vectorizer = TfidfVectorizer()

df = pd.read_csv(DATA_DIR / 'ielts_data.csv') 
df = df.iloc[:num_rows, :3]
df.columns = ['prompt', 'essay', 'score']

nlp = spacy.load("en_core_web_sm")
try:
    tool = language_tool_python.LanguageTool('en-US')
except Exception:
    tool = None

transformer = TransformerFeatures().fit()
hidden_size = transformer.model.config.hidden_size

features = []

def detect_grammar_errors(doc):
    errors = 0
    for sent in doc.sents:
        sent_tags = [token.tag_ for token in sent]
        sent_pos = [token.pos_ for token in sent]

        if not any(tag in {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'} for tag in sent_tags):
            errors += 1

        if 'IN' in sent_tags and len(sent) < 4:
            errors += 1

        if 'MD' in sent_tags and not any(tag == 'VB' for tag in sent_tags):
            errors += 1

        for i, token in enumerate(sent[:-1]):
            if token.text.lower() in {'he', 'she', 'it'} and sent[i + 1].tag_ == 'VB':
                errors += 1

        if sent.text.count(',') > 0 and sum(1 for t in sent if t.pos_ == 'VERB') >= 2 and not any(t.tag_ == 'CC' for t in sent):
            errors += 1

        pronouns = sum(1 for t in sent if t.tag_ in {'PRP', 'PRP$'})
        simple_verbs = sum(1 for t in sent if t.tag_ in {'VB', 'VBP'})
        if pronouns > 3 or simple_verbs > 3:
            errors += 1

        if sent[-1].pos_ == 'ADP':
            errors += 1

        modal_tags = [token.tag_ for token in sent if token.tag_ == 'MD']
        if len(modal_tags) > 1:
            errors += 1

        for i, token in enumerate(sent[:-1]):
            if token.tag_ == 'DT' and sent[i + 1].pos_ not in {'ADJ', 'NOUN'}:
                errors += 1

        for i in range(len(sent) - 1):
            if sent[i].is_alpha and sent[i+1].is_alpha and sent[i].text.lower() == sent[i+1].text.lower():
                errors += 1

        first_alpha = None
        for token in sent:
            if token.is_alpha:
                first_alpha = token
                break
        if first_alpha and first_alpha.text[0].islower():
            errors += 1

        for i in range(len(sent) - 1):
            if sent[i].is_punct and sent[i+1].is_punct:
                errors += 1

    return errors

def sliding_window_repeat(sentences, window_size=2):
    repeat_count = 0
    for i in range(len(sentences) - window_size + 1):
        window_words = [token.text.lower() for sent in sentences[i:i+window_size] for token in sent if token.is_alpha]
        repeat_count += len(window_words) - len(set(window_words))
    return repeat_count


total_rows = len(df)

for idx, row in df.iterrows():
    prompt = str(row['prompt'])
    essay = str(row['essay'])

    doc = nlp(essay)
    total_tokens = len([t for t in doc if t.is_alpha])
    total_sentences = len(list(doc.sents)) or 1

    
    tags = Counter([token.tag_ for token in doc])
    noun_count = sum(tags[t] for t in {'NN', 'NNS', 'NNP', 'NNPS'})
    verb_count = sum(tags[t] for t in {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'})
    adj_count = sum(tags[t] for t in {'JJ', 'JJR', 'JJS'})
    adv_count = sum(tags[t] for t in {'RB', 'RBR', 'RBS'})
    pronoun_count = sum(tags[t] for t in {'PRP', 'PRP$'})
    modal_count = sum(tags[t] for t in {'MD'})

    complexity_verb_ratio = verb_count / total_sentences
    adj_adv_ratio = (adj_count + adv_count) / total_tokens if total_tokens > 0 else 0

    
    avg_sentence_length = textstat.avg_sentence_length(essay)
    avg_syllables_per_word = textstat.avg_syllables_per_word(essay)
    flesch_score = textstat.flesch_reading_ease(essay)

    
    tfidf_matrix = tfidf_vectorizer.fit_transform([prompt, essay])
    tfidf_cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    essay_tfidf = tfidf_vectorizer.fit_transform([essay])
    tfidf_scores = essay_tfidf.toarray()[0]
    mean_tfidf_score = np.mean(tfidf_scores)
    std_tfidf_score = np.std(tfidf_scores)

    word_list = [token.text.lower() for token in doc if token.is_alpha]
    unique_word_ratio = len(set(word_list)) / len(word_list) if word_list else 0

    
    sentences = list(doc.sents)
    window_repeat_count = sliding_window_repeat(sentences, window_size=2)

    
    emb_prompt, emb_essay = transformer.transform([prompt, essay])
    bert_cos_sim = cosine_similarity([emb_prompt], [emb_essay])[0][0]
    full_essay_dims_list = emb_essay.flatten().tolist()
    essay_dims = full_essay_dims_list[:40]

    grammar_errors = detect_grammar_errors(doc)

    
    features.append([
        noun_count, verb_count, adj_count, adv_count,
        pronoun_count, modal_count,
        complexity_verb_ratio, adj_adv_ratio,
        avg_sentence_length, avg_syllables_per_word, flesch_score,
        tfidf_cos_sim,
        mean_tfidf_score, std_tfidf_score, unique_word_ratio,
        window_repeat_count, grammar_errors, bert_cos_sim,
        *essay_dims,
        row['score']   
    ])
    
    if (idx + 1) % 100 == 0 or (idx + 1) == total_rows:
        percent_complete = (idx + 1) / total_rows * 100
        print(f"Processed {idx + 1} out of {total_rows} rows ({percent_complete:.2f}%)")

n_total = len(features[0])
n_score = 1

cols = [
    'noun_count','verb_count','adj_count','adv_count',
    'pronoun_count','modal_count',
    'complexity_verb_ratio','adj_adv_ratio',
    'avg_sentence_length','avg_syllables_per_word','flesch_reading_ease',
    'tfidf_cosine_similarity','mean_tfidf_score','std_tfidf_score', 'unique_word_ratio', 'window_repeat_count',
    'grammar_error_count','bert_cosine_similarity'
]
n_fixed = len(cols)

cols += [f'bert_dim_{i}' for i in range(40)]
cols += ['score']

feature_df = pd.DataFrame(features, columns=cols)
output_path = DATA_DIR / 'extracted_features.csv'
feature_df.to_csv(output_path, index=False)
print("All features saved to extracted_features.csv")