from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

# def index(request):
#     return HttpResponse("Hello, world. This is segmentation view.")

def index(request):
    template = loader.get_template("base.html")
    return HttpResponse(template.render(request))