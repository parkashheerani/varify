from django.shortcuts import render
import pickle
from langdetect import detect
from django.contrib import messages
from grams import grams

def home(request):
    return render(request, 'index.html')

def getPredictions(news):
    model = pickle.load(open('ml_model.pkl', 'rb'))

    prediction = model.predict([news])

    return prediction[0]

def result(request):

    new_news = str(request.GET['news'])

    if detect(new_news) == 'ur':
        if len(new_news.split(' ')) > 50:
            result = getPredictions(new_news)
            return render(request, 'result.html', {'result': result})
        else:
            return render(request, 'result.html', {'result': 'News is too short'})
    else:
        return render(request, 'result.html', {'result': 'Please enter Urdu News only!'})
