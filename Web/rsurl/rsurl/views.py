# rsurl/views.py
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
import requests

def home(request):
    return render(request, 'home/index.html')

def movies(request):
    return render(request, 'home/movies.html')

def recommendations_view(request, user_id):
    url = "http://127.0.0.1:7000/recommendations"  # FastAPI URL
    payload = {"user_id": user_id}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        recommendations = data.get("recommendations", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching recommendations: {e}")
        recommendations = []

    return render(request, 'recommendations.html', {'recommendations': recommendations})
