from __future__ import print_function

import pygame
from pygame.locals import K_a
var = 1
import glob
import os
import sys
import random
import time
import cv2
import numpy

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla



def process_data(image):
    image1 = numpy.array(image.raw_data)
    image2 = numpy.reshape(image1, (300, 600, 4))
    # print(numpy.shape(image2))
    # print(image2)
    # print(var)
    global var
    var = var + 1
    a = str(var)
    # cv2.imwrite("C:/CARLA_0.9.10/WindowsNoEditor/PythonAPI/examples/output/" + a + ".png", image2)


def main():
    actorlist = []
    try:
        actor_list = []
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.load_world('Town05')

        blueprintlibrary = world.get_blueprint_library()
        vehicle_bp = blueprintlibrary.filter('cybertruck')[0]
        transform = carla.Transform(carla.Location(x=150, y=221, z=5), carla.Rotation(yaw=0))
        vehicle = world.spawn_actor(vehicle_bp, transform)
        vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
        actor_list.append(vehicle)

        camera_bp = blueprintlibrary.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '600')
        camera_bp.set_attribute('image_size_y', '300')
        camera_bp.set_attribute('fov', '105')
        print('hello')

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        # camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))
        actorlist.append(camera)

        # camera.listen(lambda data: process_data(data))
        # camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))
        print('Hello')

        time.sleep(2)
    finally:
        print("Deleted everything")
        for actor in actorlist:
            actor.destroy()


if __name__ == '__main__':
    main()
