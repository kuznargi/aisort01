from django.contrib import admin
from django.urls import path
from app.views import index,process_frame,video_feed_yolo,analysis
from app import views

urlpatterns = [
    # path('admin/', admin.site.urls),
    # path('',index,name='index'),
    # path('video/', test, name='test'),
    # path('video_feed_yolo/', video_feed_yolo, name='video_feed_yolo'),
    # path("analysis/", analysis, name="analysis"),
        path('', views.index, name='index'),
    path('video_feed_yolo/', views.video_feed_yolo, name='video_feed_yolo'),
    path('analysis/', views.analysis, name='analysis'),
    path('process_frame/', views.process_frame, name='process_frame'),
]
