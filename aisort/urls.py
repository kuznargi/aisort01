from django.contrib import admin
from django.urls import path
from app.views import index,test,video_feed_yolo,analysis

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',index,name='index'),
    path('video/', test, name='test'),
    path('video_feed_yolo/', video_feed_yolo, name='video_feed_yolo'),
    path("analysis/", analysis, name="analysis"),
]
