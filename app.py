#
# # === IMPORTS ===
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import gc
# import time
# import torch
# import warnings
# import pandas as pd
# import numpy as np
# import nltk
# import spacy
# import scispacy
#
# from flask import Flask, request, jsonify, render_template
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords, wordnet as wn
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer, util
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from IndicTransToolkit.IndicTransToolkit import IndicProcessor
#
# warnings.filterwarnings('ignore', category=FutureWarning)
#
# # === DOWNLOAD NLTK STUFF ===
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
#
# # === INITIALIZE NLP TOOLS ===
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words("english"))
# nlp = spacy.load("en_core_sci_sm")
#
# # === LOAD DATASET ===
# file_path = "Extracted_Medical_Q_A.csv"
# df = pd.read_csv(file_path)
# if "Question" not in df.columns or "Clean_Answer" not in df.columns:
#     raise KeyError("Missing 'Question' or 'Clean_Answer' column in dataset.")
# df = df.dropna(subset=["Question", "Clean_Answer"]).drop_duplicates(subset=["Question"]).reset_index(drop=True)
#
# # === LOAD EMBEDDINGS ===
# bert_model = SentenceTransformer("all-MiniLM-L6-v2")
# question_embeddings = bert_model.encode(df["Question"].tolist(), convert_to_tensor=True)
#
# # === TF-IDF VECTOR SETUP ===
# def preprocess_text(text):
#     tokens = word_tokenize(text.lower())
#     processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
#     return " ".join(processed_tokens)
#
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(df["Question"].apply(preprocess_text))
#
# # === TRANSLATION SETUP ===
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 4
#
# def initialize_model_and_tokenizer(ckpt_dir, quantization=None):
#     from transformers import BitsAndBytesConfig
#
#     if quantization == "4-bit":
#         qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
#     elif quantization == "8-bit":
#         qconfig = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_use_double_quant=True, bnb_8bit_compute_dtype=torch.bfloat16)
#     else:
#         qconfig = None
#
#     tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         ckpt_dir,
#         trust_remote_code=True,
#         low_cpu_mem_usage=True,
#         quantization_config=qconfig,
#     )
#
#     if qconfig is None:
#         model = model.to(DEVICE)
#         if DEVICE == "cuda":
#             model.half()
#
#     model.eval()
#     return tokenizer, model
#
# def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
#     translations = []
#     for i in range(0, len(input_sentences), BATCH_SIZE):
#         batch = input_sentences[i : i + BATCH_SIZE]
#         batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
#         inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
#
#         with torch.no_grad():
#             generated = model.generate(
#                 **inputs,
#                 use_cache=True,
#                 min_length=0,
#                 max_length=512,
#                 num_beams=5,
#                 num_return_sequences=1,
#                 early_stopping=True,
#                 length_penalty=1.0
#             )
#
#         with tokenizer.as_target_tokenizer():
#             decoded = tokenizer.batch_decode(
#                 generated.detach().cpu().tolist(),
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=True
#             )
#
#         translations += ip.postprocess_batch(decoded, lang=tgt_lang)
#
#         del inputs, generated
#         torch.cuda.empty_cache()
#         gc.collect()
#
#     return translations
#
# def translate_paragraph(paragraph, src_lang, tgt_lang, model, tokenizer, ip):
#     sentences = sent_tokenize(paragraph)
#     translated_sentences = batch_translate(sentences, src_lang, tgt_lang, model, tokenizer, ip)
#     return " ".join(translated_sentences)
#
# def translate_text(text, src_lang, tgt_lang, model, tokenizer, ip):
#     try:
#         return translate_paragraph(text, src_lang, tgt_lang, model, tokenizer, ip)
#     except Exception as e:
#         print(f"Translation Error: {e}")
#         return None
#
# def truncate_answer(answer, max_sentences=3):
#     import re
#     sentences = re.split(r'(?<=[.!?]) +', answer.strip())
#     return ' '.join(sentences[:max_sentences])
#
# # === MAIN RETRIEVAL FUNCTION ===
# def retrieve_best_answer(input_question_en, top_k=1, match_threshold=0.90):
#     processed_input = preprocess_text(input_question_en)
#     input_vector = vectorizer.transform([processed_input])
#     tfidf_scores = np.dot(input_vector, tfidf_matrix.T).toarray().flatten()
#     input_embedding = bert_model.encode([input_question_en], convert_to_tensor=True).view(1, -1)
#     bert_scores = util.pytorch_cos_sim(input_embedding, question_embeddings)[0].cpu().numpy()
#     final_scores = (tfidf_scores * 0.4) + (bert_scores * 0.6)
#     top_indices = np.argsort(final_scores)[-top_k:][::-1]
#
#     results = []
#     for idx in top_indices:
#         matched_q = df.iloc[idx]["Question"]
#         matched_a = df.iloc[idx]["Clean_Answer"]
#         score = final_scores[idx]
#
#         if matched_q.strip().lower() == input_question_en.strip().lower() or score >= match_threshold:
#             return [(matched_q, truncate_answer(matched_a), score)]
#
#         results.append((matched_q, truncate_answer(matched_a), score))
#
#     return results
#
# # === FLASK SETUP ===
# app = Flask(__name__)
#
# # Load models ONCE (so they don’t reload every request)
# (en_indic_tokenizer, en_indic_model) = initialize_model_and_tokenizer("ai4bharat/indictrans2-en-indic-1B")
# (indic_en_tokenizer, indic_en_model) = initialize_model_and_tokenizer("ai4bharat/indictrans2-indic-en-dist-200M")
# ip = IndicProcessor(inference=True)
#
# # === FLASK ROUTES ===
# @app.route("/")
# def index():
#     return render_template("index.html")
#
# @app.route("/query", methods=["POST"])
# def query():
#     punjabi_query = request.form.get("query", "").strip()
#     if not punjabi_query:
#         return jsonify({"error": "❌ Please enter a valid question."})
#
#     steps = {}
#
#     # Step 1: Translate Punjabi to English
#     steps["step_1"] = "🔄 Translating Punjabi query to English..."
#     english_query = translate_text(punjabi_query, "pan_Guru", "eng_Latn", indic_en_model, indic_en_tokenizer, ip)
#     if not english_query:
#         return jsonify({"error": "❌ Translation to English failed."})
#     steps["translated_query"] = english_query
#
#     # Step 2: Retrieve answer
#     steps["step_2"] = "📡 Retrieving best-matched answers..."
#     results = retrieve_best_answer(english_query, top_k=1)
#     print(results)
#
#     if not results:
#         return jsonify({"error": "❌ No relevant answer found."})
#
#     steps["step_3"] = "🔄 Translating result(s) back to Punjabi..."
#     punjabi_outputs = []
#
#     for idx, (matched_question, answer, score) in enumerate(results, 1):
#         translated_answer = translate_text(answer, "eng_Latn", "pan_Guru", en_indic_model, en_indic_tokenizer, ip)
#         if not translated_answer:
#             translated_answer = "❌ Translation error while returning result."
#         punjabi_outputs.append(f"{idx}. {translated_answer} (Score: {score:.4f})")
#
#     steps["final_results"] = punjabi_outputs
#     return jsonify(steps)
#
#
# # === START FLASK SERVER ===
# if __name__ == "__main__":
#     app.run(debug=True)
# === IMPORTS ===
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import time
import torch
import warnings
import pandas as pd
import numpy as np
import nltk
import spacy
import scispacy

from flask import Flask, request, jsonify, render_template
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.IndicTransToolkit import IndicProcessor

warnings.filterwarnings('ignore', category=FutureWarning)

# === DOWNLOAD NLTK STUFF ===
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# === INITIALIZE NLP TOOLS ===
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_sci_sm")

# === LOAD DATASET ===
file_path = "Extracted_Medical_Q_A.csv"
df = pd.read_csv(file_path)
if "Question" not in df.columns or "Clean_Answer" not in df.columns:
    raise KeyError("Missing 'Question' or 'Clean_Answer' column in dataset.")
df = df.dropna(subset=["Question", "Clean_Answer"]).drop_duplicates(subset=["Question"]).reset_index(drop=True)

# === LOAD EMBEDDINGS ===
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = bert_model.encode(df["Question"].tolist(), convert_to_tensor=True)

# === TF-IDF VECTOR SETUP ===
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(processed_tokens)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Question"].apply(preprocess_text))

# === TRANSLATION SETUP ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

def initialize_model_and_tokenizer(ckpt_dir, quantization=None):
    from transformers import BitsAndBytesConfig

    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_use_double_quant=True, bnb_8bit_compute_dtype=torch.bfloat16)
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig is None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()
    return tokenizer, model

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=512,
                num_beams=5,
                num_return_sequences=1,
                early_stopping=True,
                length_penalty=1.0
            )

        with tokenizer.as_target_tokenizer():
            decoded = tokenizer.batch_decode(
                generated.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

        translations += ip.postprocess_batch(decoded, lang=tgt_lang)

        del inputs, generated
        torch.cuda.empty_cache()
        gc.collect()

    return translations

def translate_paragraph(paragraph, src_lang, tgt_lang, model, tokenizer, ip):
    sentences = sent_tokenize(paragraph)
    translated_sentences = batch_translate(sentences, src_lang, tgt_lang, model, tokenizer, ip)
    return " ".join(translated_sentences)

def translate_text(text, src_lang, tgt_lang, model, tokenizer, ip):
    try:
        return translate_paragraph(text, src_lang, tgt_lang, model, tokenizer, ip)
    except Exception as e:
        print(f"Translation Error: {e}")
        return None

def truncate_answer(answer, max_sentences=6):
    import re
    sentences = re.split(r'(?<=[.!?]) +', answer.strip())
    return ' '.join(sentences[:max_sentences])

# === MAIN RETRIEVAL FUNCTION ===
def retrieve_best_answer(input_question_en, top_k=1, match_threshold=0.90):
    processed_input = preprocess_text(input_question_en)
    input_vector = vectorizer.transform([processed_input])
    tfidf_scores = np.dot(input_vector, tfidf_matrix.T).toarray().flatten()
    input_embedding = bert_model.encode([input_question_en], convert_to_tensor=True).view(1, -1)
    bert_scores = util.pytorch_cos_sim(input_embedding, question_embeddings)[0].cpu().numpy()
    final_scores = (tfidf_scores * 0.4) + (bert_scores * 0.6)
    top_indices = np.argsort(final_scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        matched_q = df.iloc[idx]["Question"]
        matched_a = df.iloc[idx]["Clean_Answer"]
        score = final_scores[idx]

        if matched_q.strip().lower() == input_question_en.strip().lower() or score >= match_threshold:
            return [(matched_q, truncate_answer(matched_a), score)]

        results.append((matched_q, truncate_answer(matched_a), score))

    return results

# === FLASK SETUP ===
app = Flask(__name__)

# Load models ONCE (so they don’t reload every request)
(en_indic_tokenizer, en_indic_model) = initialize_model_and_tokenizer("ai4bharat/indictrans2-en-indic-1B")
(indic_en_tokenizer, indic_en_model) = initialize_model_and_tokenizer("ai4bharat/indictrans2-indic-en-dist-200M")
ip = IndicProcessor(inference=True)

# === FLASK ROUTES ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    punjabi_query = request.form.get("query", "")
    if not punjabi_query.strip():
        return jsonify({"error": "❌ Please enter a valid question."})

    english_query = translate_text(punjabi_query, "pan_Guru", "eng_Latn", indic_en_model, indic_en_tokenizer, ip)
    if not english_query:
        return jsonify({"error": "❌ Translation to English failed."})

    results = retrieve_best_answer(english_query, top_k=1)
    punjabi_outputs = []

    for idx, (matched_question, answer, score) in enumerate(results, 1):
        translated_answer = translate_text(answer, "eng_Latn", "pan_Guru", en_indic_model, en_indic_tokenizer, ip)
        if not translated_answer:
            translated_answer = "❌ Translation error while returning result."
        punjabi_outputs.append(f"{idx}. {translated_answer} (Score: {score:.4f})")

    return jsonify({"results": punjabi_outputs})

# === START FLASK SERVER ===
if __name__ == "__main__":
    app.run(debug=True)
