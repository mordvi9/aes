import pandas as pd
import spacy
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from collections import Counter
import numpy as np
import streamlit as st

@st.cache_resource
def load_nlp_models():
    nlp = spacy.load("en_core_web_sm")    
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    return nlp, bert_tokenizer, bert_model

tfidf_vectorizer = TfidfVectorizer()

def sliding_window_repeat(sentences, window_size=2):
    repeat_count = 0
    for i in range(len(sentences) - window_size + 1):
        window_words = [token.text.lower() for sent in sentences[i:i+window_size] for token in sent if token.is_alpha]
        repeat_count += len(window_words) - len(set(window_words))
    return repeat_count


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

def extract_all_features(prompt, essay, fitted_tfidf, nlp, tokenizer, model):
    doc = nlp(essay)
    total_tokens = len([t for t in doc if t.is_alpha]) or 1
    total_sentences = len(list(doc.sents)) or 1

    tags = Counter([token.tag_ for token in doc])
    noun_count = sum(tags[t] for t in {'NN', 'NNS', 'NNP', 'NNPS'})
    verb_count = sum(tags[t] for t in {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'})
    adj_count = sum(tags[t] for t in {'JJ', 'JJR', 'JJS'})
    adv_count = sum(tags[t] for t in {'RB', 'RBR', 'RBS'})
    pronoun_count = sum(tags[t] for t in {'PRP', 'PRP$'})
    modal_count = sum(tags[t] for t in {'MD'})

    avg_sentence_length = textstat.avg_sentence_length(essay)
    avg_syllables_per_word = textstat.avg_syllables_per_word(essay)
    flesch_score = textstat.flesch_reading_ease(essay)

    tfidf_matrix = fitted_tfidf.transform([prompt, essay])
    tfidf_cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    essay_tfidf_matrix = fitted_tfidf.transform([essay])
    essay_scores = essay_tfidf_matrix.toarray()[0]
    non_zero_scores = essay_scores[np.nonzero(essay_scores)]
    mean_tfidf_score = np.mean(non_zero_scores) if len(non_zero_scores) > 0 else 0
    std_tfidf_score = np.std(non_zero_scores) if len(non_zero_scores) > 0 else 0

    word_list = [token.text.lower() for token in doc if token.is_alpha]
    unique_word_ratio = len(set(word_list)) / len(word_list) if word_list else 0
    
    sentences = list(doc.sents)
    window_repeat_count = sliding_window_repeat(sentences, window_size=2)
    
    with torch.no_grad():
        prompt_enc = tokenizer(prompt, padding=True, truncation=True, max_length=512, return_tensors='pt')
        prompt_out = model(prompt_enc['input_ids'], attention_mask=prompt_enc['attention_mask'])
        emb_prompt = prompt_out.last_hidden_state[:,0,:].cpu().numpy()

        essay_enc = tokenizer(essay, padding=True, truncation=True, max_length=512, return_tensors='pt')
        essay_out = model(essay_enc['input_ids'], attention_mask=essay_enc['attention_mask'])
        emb_essay = essay_out.last_hidden_state[:,0,:].cpu().numpy()
        emb_essay_chopped = emb_essay[:, :40]

    bert_cos_sim = cosine_similarity(emb_prompt, emb_essay)[0][0]

    grammar_errors = detect_grammar_errors(doc)

    feature_dict = {
        'noun_count': noun_count, 'verb_count': verb_count, 'adj_count': adj_count, 'adv_count': adv_count,
        'pronoun_count': pronoun_count, 'modal_count': modal_count,
        'complexity_verb_ratio': verb_count / total_sentences,
        'adj_adv_ratio': (adj_count + adv_count) / total_tokens,
        'avg_sentence_length': avg_sentence_length, 'avg_syllables_per_word': avg_syllables_per_word,
        'flesch_reading_ease': flesch_score,
        'tfidf_cosine_similarity': tfidf_cos_sim, 'mean_tfidf_score': mean_tfidf_score,
        'std_tfidf_score': std_tfidf_score, 'unique_word_ratio': unique_word_ratio,
        'window_repeat_count': window_repeat_count, 'grammar_error_count': grammar_errors,
        'bert_cosine_similarity': bert_cos_sim
    }

    for i, dim in enumerate(emb_essay_chopped.flatten()):
        feature_dict[f'bert_dim_{i}'] = dim
        
    return feature_dict