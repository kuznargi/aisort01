from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed_yolo/', views.video_feed_yolo, name='video_feed_yolo'),
    path('analysis/', views.analysis, name='analysis'),
    path('process_frame/', views.process_frame, name='process_frame'),
]
