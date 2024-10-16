#!/usr/bin/env python
# coding: utf-8


from transformers import BertModel, BertTokenizer, BertConfig
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import transformers
import torch
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import sklearn.metrics as metrics
import nltk
from oauth2client.client import GoogleCredentials
from google.colab import auth
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
import spacy
import random
from tqdm import tqdm
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import math
get_ipython().system('pip install scipy')
get_ipython().system('pip install torch torchvision torchaudio')


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")


plt.style.use('seaborn')


tqdm.pandas()


get_ipython().system('pip install PyDrive')


# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Download the file from Google Drive.
# Make sure you have replaced '<REPLACE_WITH_FILE_ID>' with the correct file ID.
downloaded = drive.CreateFile({'id': '1C6Dj2zN0Gm5vQWH2VVrDSOkVsXkN3WY4'})
downloaded.GetContentFile('imdb_reviews.tsv')

# Load the downloaded file into a pandas DataFrame.
df_reviews = pd.read_csv('imdb_reviews.tsv', sep='\t',
                         dtype={'votes': 'Int64'})
corpus = df_reviews['review']

# Initialize spaCy (without 'parser' and 'ner').
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# In[6]:


print(df_reviews.info())
print(df_reviews.describe())
print(df_reviews['review'].head())


# Distribución de Clases
class_counts = df_reviews['pos'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Distribución de Clases')
plt.xlabel('Sentimiento')
plt.ylabel('Número de Reseñas')
plt.xticks(ticks=[0, 1], labels=['Negativo', 'Positivo'])
plt.show()


# Procesar texto: Limpiar Expresiones regulares
def clear_text(text):
    clean_text = re.sub(r'[^\w\s]', '', text)
    clean_text = re.sub(r'[^a-zA-Z\']', ' ', clean_text)
    clean_text = re.sub(r'\d+', '', clean_text)
    clean_text = " ".join(clean_text.split())
    return clean_text


# Lematizar

def lemmatize(doc):
    return ' '.join([token.lemma_ for token in doc])

# Procesamiento por lotes optimizado


def batch_process(corpus, batch_size=100):  # Reducir el tamaño del lote
    processed_texts = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="Procesando por lotes"):
        batch = corpus[i:i+batch_size]
        # Procesar en un solo hilo para evitar sobrecarga
        docs = list(nlp.pipe(batch, batch_size=batch_size, n_process=1))
        processed_texts.extend([lemmatize(doc) for doc in docs])
    return processed_texts


# Procesar el corpus
corpus_processed = batch_process(corpus, batch_size=100)


# Vectorizar con TF-IDF

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words(
    'english')  # Use a list instead of a set
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000)
tf_idf = vectorizer.fit_transform(corpus_processed)

print('El tamaño de la matriz TF-IDF:', tf_idf.shape)


fig, axs = plt.subplots(2, 1, figsize=(16, 8))

ax = axs[0]

dft1 = df_reviews[['tconst', 'start_year']].drop_duplicates(
)['start_year'].value_counts().sort_index()
dft1 = dft1.reindex(index=np.arange(
    dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)
dft1.plot(kind='bar', ax=ax)
ax.set_title('Número de películas a lo largo de los años')

ax = axs[1]

dft2 = df_reviews.groupby(['start_year', 'pos'])['pos'].count().unstack()
dft2 = dft2.reindex(index=np.arange(
    dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)

dft2.plot(kind='bar', stacked=True, label='#reviews (neg, pos)', ax=ax)

dft2 = df_reviews['start_year'].value_counts().sort_index()
dft2 = dft2.reindex(index=np.arange(
    dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
dft3 = (dft2/dft1).fillna(0)
axt = ax.twinx()
dft3.reset_index(drop=True).rolling(5).mean().plot(
    color='orange', label='reviews per movie (avg over 5 years)', ax=axt)

lines, labels = axt.get_legend_handles_labels()
ax.legend(lines, labels, loc='upper left')

ax.set_title('Número de reseñas a lo largo de los años')

fig.tight_layout()


fig, axs = plt.subplots(1, 2, figsize=(16, 5))

ax = axs[0]
dft = df_reviews.groupby('tconst')['review'].count() \
    .value_counts() \
    .sort_index()
dft.plot.bar(ax=ax)
ax.set_title('Gráfico de barras de #Reseñas por película')

ax = axs[1]
dft = df_reviews.groupby('tconst')['review'].count()
sns.kdeplot(dft, ax=ax)
ax.set_title('Gráfico KDE de #Reseñas por película')

fig.tight_layout()


df_reviews['pos'].value_counts()


fig, axs = plt.subplots(1, 2, figsize=(12, 4))

ax = axs[0]
dft = df_reviews.query('ds_part == "train"')[
    'rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1),
                  max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('El conjunto de entrenamiento: distribución de puntuaciones')

ax = axs[1]
dft = df_reviews.query('ds_part == "test"')[
    'rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1),
                  max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('El conjunto de prueba: distribución de puntuaciones')

fig.tight_layout()


fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(
    width_ratios=(2, 1), height_ratios=(1, 1)))

ax = axs[0][0]

dft = df_reviews.query('ds_part == "train"').groupby(
    ['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(
    dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title(
    'El conjunto de entrenamiento: número de reseñas de diferentes polaridades por año')

ax = axs[0][1]

dft = df_reviews.query('ds_part == "train"').groupby(
    ['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title(
    'El conjunto de entrenamiento: distribución de diferentes polaridades por película')

ax = axs[1][0]

dft = df_reviews.query('ds_part == "test"').groupby(
    ['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(
    dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title(
    'El conjunto de prueba: número de reseñas de diferentes polaridades por año')

ax = axs[1][1]

dft = df_reviews.query('ds_part == "test"').groupby(
    ['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title(
    'El conjunto de prueba: distribución de diferentes polaridades por película')

fig.tight_layout()


def evaluate_model(model, train_features, train_target, test_features, test_target):

    eval_stats = {}

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):

        eval_stats[type] = {}

        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]

        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba >= threshold)
                     for threshold in f1_thresholds]

        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps

        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={
                f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx],
                    color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1')

        # ROC
        ax = axs[1]
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx],
                    color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')
        ax.set_title(f'Curva ROC')

        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx],
                    color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(
            target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)

    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(
        index=('Exactitud', 'F1', 'APS', 'ROC AUC'))

    print(df_eval_stats)

    return


def preprocess_text_series(series):
    return series.str.lower().replace(r'\d+', '', regex=True).replace(r'[^\w\s]', '', regex=True)


df_reviews['review_norm'] = preprocess_text_series(df_reviews['review'])


df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

train_target = df_reviews_train['pos'].values
test_target = df_reviews_test['pos'].values

print(df_reviews_train.shape)
print(df_reviews_test.shape)


tfidf_vectorizer_1 = TfidfVectorizer(stop_words=stop_words)

train_features_1 = tfidf_vectorizer_1.fit_transform(
    df_reviews_train['review_norm'])
test_features_1 = tfidf_vectorizer_1.transform(df_reviews_test['review_norm'])


model_0 = DummyClassifier(strategy="most_frequent")


model_0.fit(train_features_1, train_target)

# Evaluar el modelo
evaluate_model(model_0, train_features_1, train_target,
               test_features_1, test_target)


nltk.download('stopwords')
stop_words = stopwords.words('english')

tfidf_vectorizer_2 = TfidfVectorizer(stop_words=stop_words)

train_features = tfidf_vectorizer_2.fit_transform(
    df_reviews_train['review_norm'])
test_features = tfidf_vectorizer_2.transform(df_reviews_test['review_norm'])

model_2 = LogisticRegression()

# Entrenar el modelo
model_2.fit(train_features, train_target)

# Evaluar el modelo


def evaluate_model(model_2, train_features, train_target, test_features, test_target):
    train_predictions = model_2.predict(train_features)
    test_predictions = model_2.predict(test_features)

    metrics = {
        'Exactitud': accuracy_score(test_target, test_predictions),
        'F1': f1_score(test_target, test_predictions),
        'APS': average_precision_score(test_target, test_predictions),
        'ROC AUC': roc_auc_score(test_target, model_2.predict_proba(test_features)[:, 1])
    }

    for metric, score in metrics.items():
        print(f'{metric} - Train: {accuracy_score(train_target,
              model_2.predict(train_features)):.2f}, Test: {score:.2f}')


evaluate_model(model_2, train_features, train_target,
               test_features, test_target)


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])


def text_preprocessing_3(texts, batch_size=10000, n_process=4):
    # Usar n_process para procesamiento paralelo
    tokens = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        # Lematización
        tokens.append(
            ' '.join([token.lemma_ for token in doc if not token.is_stop]))
    return tokens


# Preprocesamiento de textos
train_processed_3 = text_preprocessing_3(df_reviews_train['review_norm'])
test_processed_3 = text_preprocessing_3(df_reviews_test['review_norm'])

# Vectorización TF-IDF
tfidf_vectorizer_3 = TfidfVectorizer(
    stop_words=stop_words, ngram_range=(1, 2), max_df=0.9, min_df=5)
train_features_3 = tfidf_vectorizer_3.fit_transform(train_processed_3)
test_features_3 = tfidf_vectorizer_3.transform(test_processed_3)

model_3 = LogisticRegression(max_iter=100, solver='liblinear', C=0.5)

# Entrenar el modelo
model_3.fit(train_features_3, train_target)


# Evaluar el modelo
def evaluate_model(model, train_features, train_target, test_features, test_target):
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    metrics = {
        'Exactitud': accuracy_score(test_target, test_predictions),
        'F1': f1_score(test_target, test_predictions),
        'APS': average_precision_score(test_target, test_predictions),
        'ROC AUC': roc_auc_score(test_target, model.predict_proba(test_features)[:, 1])
    }

    for metric, score in metrics.items():
        print(f'{metric} - Train: {accuracy_score(train_target,
              model.predict(train_features)):.2f}, Test: {score:.2f}')


# Llamar a la función de evaluación
evaluate_model(model_3, train_features_3, train_target,
               test_features_3, test_target)


def text_preprocessing_4(texts, batch_size=10000, n_process=4):
    # Usar n_process para procesamiento paralelo
    tokens = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        # Lematización
        tokens.append(
            ' '.join([token.lemma_ for token in doc if not token.is_stop]))
    return tokens


# Preprocesamiento de textos
train_processed_4 = text_preprocessing_4(df_reviews_train['review_norm'])
test_processed_4 = text_preprocessing_4(df_reviews_test['review_norm'])


# Vectorización TF-IDF
tfidf_vectorizer_4 = TfidfVectorizer(
    stop_words=stop_words, ngram_range=(1, 2), max_df=0.9, min_df=5)
train_features_4 = tfidf_vectorizer_4.fit_transform(train_processed_4)
test_features_4 = tfidf_vectorizer_4.transform(test_processed_4)

model_4 = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100)

# Entrenar el modelo
model_4.fit(train_features_4, train_target)

# Evaluar el modelo


def evaluate_model_lgbm(model, train_features, train_target, test_features, test_target):
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    metrics = {
        'Exactitud': accuracy_score(test_target, test_predictions),
        'F1': f1_score(test_target, test_predictions),
        'APS': average_precision_score(test_target, test_predictions),
        'ROC AUC': roc_auc_score(test_target, model.predict_proba(test_features)[:, 1])
    }

    for metric, score in metrics.items():
        print(f'{metric} - Train: {accuracy_score(train_target,
              model.predict(train_features)):.2f}, Test: {score:.2f}')


# Llamar a la función de evaluación
evaluate_model_lgbm(model_4, train_features_4, train_target,
                    test_features_4, test_target)


# Cargar el tokenizer y el modelo preentrenado
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
config = transformers.BertConfig.from_pretrained('bert-base-uncased')
model_9 = transformers.BertModel.from_pretrained('bert-base-uncased')

# Definir un modelo con una capa de clasificación encima de BERT


class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_labels=2):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Tomamos el token [CLS]
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


# Crear una instancia del modelo con clasificación
model_9_with_classifier = BertClassifier(model_9)

# Función para convertir texto a embeddings usando BERT


def BERT_text_to_embeddings(texts, max_length=512, batch_size=100, force_device=None, disable_progress_bar=False):
    # Convertir texto a ids de tokens y máscaras de atención
    ids_list = []
    attention_mask_list = []

    for text in texts:
        tokens = tokenizer.encode_plus(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        ids_list.append(tokens['input_ids'].squeeze().tolist())
        attention_mask_list.append(tokens['attention_mask'].squeeze().tolist())

    if force_device is not None:
        device = torch.device(force_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_9_with_classifier.to(device)
    if not disable_progress_bar:
        print(f'Uso del dispositivo {device}.')

    # Crear DataLoader para procesamiento por lotes
    ids_tensor = torch.LongTensor(ids_list)
    attention_mask_tensor = torch.LongTensor(attention_mask_list)
    dataset = TensorDataset(ids_tensor, attention_mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []

    model_9_with_classifier.eval()
    with torch.no_grad():
        for ids_batch, attention_mask_batch in tqdm(dataloader, disable=disable_progress_bar):
            ids_batch = ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)

            # Obtener logits en lugar de embeddings
            logits = model_9_with_classifier(
                input_ids=ids_batch, attention_mask=attention_mask_batch)
            # Convertir logits a probabilidades
            probs = torch.softmax(logits, dim=1)
            embeddings.append(probs.detach().cpu().numpy())

    return np.concatenate(embeddings)


train_features_9 = BERT_text_to_embeddings(
    df_reviews_train['review_norm'].tolist())


print(df_reviews_train['review_norm'].shape)
print(train_features_9.shape)
print(train_target.shape)


# Tokenización y generación de tensores para el entrenamiento y prueba
def tokenize_texts(texts, tokenizer, max_len=128):
    tokens = tokenizer(texts, max_length=max_len,
                       truncation=True, padding=True, return_tensors='pt')
    return tokens['input_ids'], tokens['attention_mask']


# Tokenizar textos de prueba
test_input_ids, test_attention_mask = tokenize_texts(
    df_reviews_test['review_norm'].tolist(), tokenizer)

# Crear dataset y dataloader
test_dataset = TensorDataset(
    test_input_ids, test_attention_mask, torch.tensor(test_target))
test_dataloader = DataLoader(test_dataset, batch_size=16)

# Evaluar el modelo en el conjunto de prueba


def evaluate_model(model, dataloader):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(
                'cuda'), attention_mask.to('cuda'), labels.to('cuda')
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            targets.extend(labels.cpu().numpy())
    return np.array(predictions), np.array(targets)


# Evaluar en conjunto de prueba
test_predictions, test_targets = evaluate_model(
    model_9_with_classifier, test_dataloader)


# Calcular métricas para prueba
test_acc = accuracy_score(test_targets, test_predictions)
test_f1 = f1_score(test_targets, test_predictions, average='weighted')
test_aps = average_precision_score(test_targets, test_predictions)
test_roc_auc = roc_auc_score(test_targets, test_predictions)


# Imprimir resultados
print(f'Exactitud - Test: {test_acc}')
print(f'F1 - Test: {test_f1}')
print(f'APS - Test: {test_aps}')
print(f'ROC AUC - Test: {test_roc_auc}')


get_ipython().system('pip install spacy')

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# puedes eliminar por completo estas reseñas y probar tus modelos en tus propias reseñas; las que se muestran a continuación son solo ejemplos

my_reviews = pd.DataFrame([
    'I did not simply like it, not my kind of movie.',
    'Well, I was bored and felt asleep in the middle of the movie.',
    'I was really fascinated with the movie',
    'Even the actors looked really old and disinterested, and they got paid to be in the movie. What a soulless cash grab.',
    'I didn\'t expect the reboot to be so good! Writers really cared about the source material',
    'The movie had its upsides and downsides, but I feel like overall it\'s a decent flick. I could see myself going to see it again.',
    'What a rotten attempt at a comedy. Not a single joke lands, everyone acts annoying and loud, even kids won\'t like this!',
    'Launching on Netflix was a brave move & I really appreciate being able to binge on episode after episode, of this exciting intelligent new drama.'
], columns=['review'])

"""
my_reviews = pd.DataFrame([
    'Simplemente no me gustó, no es mi tipo de película.',
    'Bueno, estaba aburrido y me quedé dormido a media película.',
    'Estaba realmente fascinada con la película',
    'Hasta los actores parecían muy viejos y desinteresados, y les pagaron por estar en la película. Qué robo tan desalmado.',
    '¡No esperaba que el relanzamiento fuera tan bueno! Los escritores realmente se preocuparon por el material original',
    'La película tuvo sus altibajos, pero siento que, en general, es una película decente. Sí la volvería a ver',
    'Qué pésimo intento de comedia. Ni una sola broma tiene sentido, todos actúan de forma irritante y ruidosa, ¡ni siquiera a los niños les gustará esto!',
    'Fue muy valiente el lanzamiento en Netflix y realmente aprecio poder seguir viendo episodio tras episodio de este nuevo drama tan emocionante e inteligente.'
], columns=['review'])
"""


def clear_text(text):
    # your code here
    return text


def lemmatize(doc):
    return ' '.join([token.lemma_ for token in doc])


my_reviews['review_norm'] = my_reviews['review'].apply(
    lambda x: lemmatize(nlp(clear_text(x))))

my_reviews


texts = my_reviews['review_norm']

my_reviews_pred_prob = model_2.predict_proba(
    tfidf_vectorizer_2.transform(texts))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')

texts = my_reviews['review_norm']

# Ensure text_preprocessing_3 returns a string
my_reviews_pred_prob = model_3.predict_proba(tfidf_vectorizer_3.transform(
    texts.apply(lambda x: " ".join(text_preprocessing_3(x)))))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


texts = my_reviews['review_norm']

tfidf_vectorizer_4 = tfidf_vectorizer_3
# Join the tokens returned by text_preprocessing_3 into a single string
my_reviews_pred_prob = model_4.predict_proba(tfidf_vectorizer_4.transform(
    texts.apply(lambda x: " ".join(text_preprocessing_3(x)))))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')

# Generar las probabilidades de predicción

texts = my_reviews['review_norm']
my_reviews_features_9 = BERT_text_to_embeddings(
    texts, disable_progress_bar=True)

# Obtener las probabilidades de clasificación
my_reviews_pred_prob = my_reviews_features_9[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# ## Conclusiones

# Después de llevar a cabo la ejecución de los diferentes modelos se obtuvieron los siguientes resultados:
#
# - Modelo 0 - DummyClassifier (Estrategia: "most_frequent")
#
# Exactitud: Train: 0.50, Test: 0.50
#
# F1: Train: 0.50, Test: 0.00
#
# APS: Train: 0.50, Test: 0.50
#
# ROC AUC: Train: 0.50, Test: 0.50
#
# El modelo DummyClassifier se basa en una estrategia de clase mayoritaria,
# actúa como un modelo de referencia básico. Debido a que siempre predice la clase más frecuente,
# su rendimiento es inadecuado para la tarea, especialmente en el conjunto de prueba donde no logra identificar críticas negativas.
# Además el valor de F1 es mucho menor al que se necesita en el proyecto de 0.85

# - Modelo 2 - TF-IDF y Logistic Regression
#
# Exactitud: Train: 0.94, Test: 0.88
#
# F1: Train: 0.94, Test: 0.88
#
# APS: Train: 0.94, Test: 0.84
#
# ROC AUC: Train: 0.94, Test: 0.95
#
# El modelo TF-IDF y Logistic Regression muestra un rendimiento sólido tanto en entrenamiento como en prueba.
# La alta exactitud y el alto F1 (mayor a 0.85) en ambos conjuntos de datos sugieren que el modelo de regresión logística con
# TF-IDF como vectorizador es eficaz para clasificar las reseñas. Los resultados indican que el modelo está bien ajustado y
# generaliza bien a nuevos datos.

# - Modelo 3 - spaCy, TF-IDF y Logistic Regression
#
# Exactitud: Train: 0.92, Test: 0.87
#
# F1: Train: 0.92, Test: 0.87
#
# APS: Train: 0.92, Test: 0.82
#
# ROC AUC: Train: 0.92, Test: 0.94
#
# La integración de spaCy para lematización en este modelo parece haber mantenido un rendimiento similar al del Modelo 2,
# con una ligera caída en la exactitud y en las métricas F1. La lematización no parece haber mejorado significativamente el
# rendimiento en comparación con el modelo anterior.

# - Modelo 4 - spaCy, TF-IDF y LGBMClassifier
#
# Exactitud: Train: 0.87, Test: 0.83
#
# F1: Train: 0.87, Test: 0.84
#
# APS: Train: 0.87, Test: 0.78
#
# ROC AUC: Train: 0.87, Test: 0.92
#
# El modelo LGBMClassifier muestra un buen rendimiento general, aunque ligeramente inferior al de los modelos anteriores en
# términos de exactitud y F1. Sin embargo, el ROC AUC es bastante alto, lo que indica que el modelo tiene una buena capacidad
# para distinguir entre las clases, pero puede no estar capturando todos los matices de las críticas negativas tan eficazmente
# como los modelos basados en regresión logística.

# - Modelo 9 - BERT
#
# Exactitud: Test: 0.50
#
# F1: Test: 0.39
#
# APS: Test: 0.50
#
# ROC AUC: Test: 0.50
#
# El modelo basado en BERT tiene un rendimiento muy bajo, con métricas cercanas a las del modelo Dummy.

# Mejores Modelos: Los Modelos 2 y 3, basados en TF-IDF y Logistic Regression, muestran el mejor rendimiento en términos de
# exactitud y métricas F1. Por lo cual son los que alcanzan el objetivo de detectar críticas negativas, logrando los requisitos
# del valor F1 de al menos 0.85.
#
# Modelo LGBM: Aunque es eficaz y ofrece buenas métricas, el modelo LGBMClassifier no supera a los modelos basados
# en Logistic Regression en términos de exactitud y F1. Puede ser útil si se necesita una mejor capacidad para manejar grandes
# volúmenes de datos.
