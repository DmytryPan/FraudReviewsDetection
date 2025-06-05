from django.shortcuts import render
from django.http import JsonResponse
import os
import numpy as np
import nltk
nltk.data.path.append(os.environ.get('NLTK_DATA'))
print("Текущая директория:", os.getcwd())
print("Список директорий:", os.listdir())
# print("Путь к nltk: ", os.path.exists("/opt/render/nltk_data"))
# NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nltk_data')

# if not(os.path.exists("/opt/render/nltk_data")):
#     nltk.download('stopwords')
#     nltk.download('punkt')
#     nltk.download('punkt_tab')



from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import os
import re
import torch
from torch import nn
import contractions 
import re

# класс модели-классификатора
class lstm_parallel_cnn(nn.Module):
   def __init__(self, vocab_size, embedding_matrix, embedding_dim, hidden_dim, num_filters, kernels_sizes, output_dim, dropout_rate, num_layers):
       super(lstm_parallel_cnn, self).__init__()
       embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
       self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
       self.dropout = nn.Dropout(dropout_rate)
       self.lstm = nn.LSTM(input_size=embedding_dim, 
                           hidden_size=hidden_dim, 
                           batch_first=True,
                           bidirectional=True,
                           num_layers=num_layers, 
                           dropout=dropout_rate if num_layers > 1 else 0,
                           )
       # Список сверточных слоев
       self.conv_lst = nn.ModuleList([
           nn.Sequential(
               nn.Conv1d(in_channels=hidden_dim*2,out_channels=num_filters,kernel_size=ks),
               nn.BatchNorm1d(num_filters),
               nn.ReLU(),
               nn.Dropout(dropout_rate),
               nn.AdaptiveMaxPool1d(1),
               nn.Flatten()
                        ) for ks in kernels_sizes])
       
       self.bn = nn.BatchNorm1d(num_filters)
       self.relu = torch.nn.ReLU()
       
       self.fc1 = nn.Linear(num_filters*len(kernels_sizes), output_dim) 
   def forward(self, x):
       embedded = self.dropout(self.embedding(x))
       lstm_out,_ = self.lstm(embedded)
       lstm_out = lstm_out.permute(0, 2, 1)
       conv_outs = [conv(lstm_out) for conv in self.conv_lst]
       conv_out_cat = torch.cat(conv_outs, dim=1)
       out = self.fc1(conv_out_cat)
       return out

# функция предобработки текста отзыва
def preprocess_review(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\d'\s]", '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# функция векторизации текста
def text_to_tensor(text, vocabulary, max_len=256):
    cl_text = preprocess_review(text)
    tokens = cl_text.split()[:max_len] # берем токены
    inds = [vocabulary.get(tok, vocabulary['UKN']) for tok in tokens] # вычисляем их индексы
    padding_length = max_len - len(inds)
    inds += [vocabulary['PAD']] * padding_length
    return torch.tensor(inds).unsqueeze(0)

def predict(model, text):
    model.eval()
    with torch.no_grad():
        input = text_to_tensor(text, vocabulary)
        output = model(input)
        prob = torch.sigmoid(output).item()
        label = 1 if prob >= 0.5 else 0
        return label

stop_words = set(stopwords.words('english')) # стоп-слова

# print(os.getcwd())
vocabulary = {}
with open('classifier/embeddings/vocabulary.pkl', 'rb') as f:
    vocabulary =  pickle.load(f) # отображение слово - индекс в матрице

# print(len(vocabulary))

embeddings_matrix = np.load('classifier/embeddings/embedding_matrix.npy') # матрица вложений слов

# Параметры модели
embedding_dim = 300
hidden_dim = 256
num_layers = 1
kernels_sizes = [2,3,4]
num_filters = 256
output_dim = 1

# создание и загрузка модели
model = lstm_parallel_cnn(len(vocabulary), embeddings_matrix, embedding_dim, hidden_dim, num_filters, kernels_sizes, output_dim, 0.25, 1)
model.load_state_dict(torch.load('classifier/weights/lstm_parallel_cnn.pth',  weights_only=True, map_location=torch.device('cpu')))
print('модель загружена')

# predict(model, "test text example")

def classify_review(request):
    if request.method =='POST':
        review = request.POST.get('review')
        if not review:
            return JsonResponse({'error': 'Текст не передан'}, status=400)
        label = predict(model, review)
        result = {
            'prediction': 'Реальный отзыв' if label == 1 else 'Фейковый отзыв'
        }
        return JsonResponse(result)
    return render(request, 'classifier/classify_review.html')
        



