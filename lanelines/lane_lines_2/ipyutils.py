from moviepy import editor # import VideoFileClip
from IPython import display # import HTML
import cv2

def process_image(image):
    image = cv2.flip(image,0)
    return image

def html_embed(videofile, width=500, height=400):
    return display.HTML("""
                <video width="{w}" height="{h}" controls>
                  <source src="{0}">
                </video>
                """.format(videofile,w=width, h=height))
