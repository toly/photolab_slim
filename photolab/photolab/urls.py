"""photolab URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import os
from django.contrib import admin
from django.urls import path
from django.shortcuts import render
from django.conf import settings
from django.conf.urls.static import  static

MEDIA_ROOT = settings.MEDIA_ROOT

from app.core import load_image, do_thin

import tensorflow as tf
graph = tf.get_default_graph()


def view(request):

    context = {}

    if request.method == 'POST':
        photo = request.FILES['photo']
        name = photo.name

        filename = os.path.join(MEDIA_ROOT, name)
        filename_result = os.path.join(MEDIA_ROOT, '_' + name)

        url = '/media/' + name
        url_result = '/media/' + '_' + name

        with open(filename, 'wb+') as f:
            for chunk in photo.chunks():
                f.write(chunk)

        image = load_image(filename)

        global graph
        with graph.as_default():
            image_result = do_thin(image, ratio=0.85)
            image_result.save(filename_result)

        context.update({
            'input': url,
            'output': url_result
        })

    return render(request, 'index.html', context=context)


urlpatterns = [
    path('', view),
    path('admin/', admin.site.urls),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
