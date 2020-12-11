import os
from os.path import isfile, join
import sys

import numpy as np
import pygame
import pygame.camera
from pygame.locals import *
import face_recognition

pygame.init()
pygame.camera.init()
pygame.font.init()

known_face_encodings = []
known_face_names = []
img_names = [f for f in os.listdir("images/") if isfile(join("images/", f))]
for img_name in img_names:
    img = face_recognition.load_image_file(f"images/{img_name}")
    enc = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(enc)
    known_face_names.append(img_name.split(".")[0])

print("Encoding(s) loaded")


def recognize_faces(snapshot):
    frame = pygame.surfarray.pixels3d(snapshot)
    flipped_frame = np.swapaxes(frame, 0, 1)
    face_locations = face_recognition.face_locations(flipped_frame)
    face_encodings = face_recognition.face_encodings(flipped_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[int(best_match_index)]

        face_names.append(name)
    return frame, face_locations, face_names


class Capture(object):
    def __init__(self):
        self.size = (640, 480)
        self.display = pygame.display.set_mode(self.size)
        self.clist = pygame.camera.list_cameras()
        if not self.clist:
            raise ValueError("No cameras")
        self.cam = pygame.camera.Camera(self.clist[0], self.size)
        self.cam.start()
        self.snapshot = pygame.surface.Surface(self.size, 0, self.display)

    def get_and_flip(self):
        snpsht = self.cam.get_image(self.snapshot)
        to_show, face_locations, face_names = recognize_faces(snpsht)
        pygame.surfarray.blit_array(self.display, to_show)
        myfont = pygame.font.SysFont('Comic Sans MS', 25)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            width = right-left
            y2 = round(top+(bottom-top)*0.7)
            pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(left, top, width, bottom-top), 2)
            pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(left, y2, width, round((bottom-top)*0.3)))
            textsurface = myfont.render(name, False, (0, 0, 0))
            self.display.blit(textsurface, (left, y2))
        pygame.display.flip()

    def main(self):
        while True:
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    self.cam.stop()
                    pygame.quit()
            self.get_and_flip()


c = Capture()
c.main()
