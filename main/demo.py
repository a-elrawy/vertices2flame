#!/usr/bin/env python
import os
import cv2
import torch
import numpy as np
import pickle
import time

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import tempfile
from subprocess import call

if 'DISPLAY' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pyrender
import trimesh
from psbody.mesh import Mesh
from dataset import convert_to_vertices
from models.flame_inverter import FlameInverter


# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0, bg_black=True):
    camera_params = {'c': np.array([400, 400]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    
    intensity = 2.0

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )


    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)
  
    if bg_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def main():

    inverter_path = 'checkpoints/inverter.pth'
    inverter = FlameInverter(load_path=inverter_path)
    inverter = inverter.cuda()
    demo_npy_save_folder = 'demo_npy'
    file_name = 'demo_input/00001.npy'

    inverter.eval()

    save_folder = demo_npy_save_folder

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    test(inverter, file_name,  save_folder)


def test(inverter, file_name, save_folder):
    # generate the flame params (exp and pose) for the input npy file 
    print('Generating flame params for {}...'.format(file_name))
    
    template_file = 'dataset/model/templates.pkl'
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')


    test_name = os.path.basename(file_name).split(".")[0]
    if not os.path.exists(os.path.join(save_folder,test_name)):
        os.makedirs(os.path.join(save_folder,test_name))

    # Load vertices in file name
    with open(file_name, 'rb') as fin:
        vertices = pickle.load(fin,encoding='latin1')
    
    vertices = torch.from_numpy(vertices).cuda()

    predicted_flame_exp = os.path.join(save_folder, test_name, '_exp.npy')
    predicted_flame_pose = os.path.join(save_folder, test_name, '_pose.npy')
    predicted_vertices_path = os.path.join(save_folder, test_name, '_vertices.npy')

    with torch.no_grad():
        pose, exp = inverter(prediction)
        vertices_out = convert_to_vertices(exp,pose)
        prediction = prediction.squeeze() # (seq_len, V*3)
        exp = exp.squeeze() # (seq_len, 50)
        pose = pose.squeeze() # (seq_len, 6)
        np.save(predicted_vertices_path, prediction.detach().cpu().numpy())
        np.save(predicted_flame_exp, exp.detach().cpu().numpy())
        np.save(predicted_flame_pose, pose.detach().cpu().numpy())
        print(f'Save facial animation in {predicted_vertices_path}')

    ######################################################################################
    ##### render the npy file


    print("rendering: ", test_name)
    template_file = 'dataset/model/FLAME_sample.ply'

    template = Mesh(filename=template_file)
    predicted_vertices = np.load(predicted_vertices_path)
    predicted_vertices = np.reshape(predicted_vertices,(-1, 5023, 3))

    output_path = os.path.join(save_folder, test_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_frames = predicted_vertices.shape[0]
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)
    
    writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (800, 800), True)
    center = np.mean(predicted_vertices[0], axis=0)
    
    for i_frame in range(num_frames):
        render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        pred_img = render_mesh_helper(render_mesh, center)
        pred_img = pred_img.astype(np.uint8)
        writer.write(pred_img)

    writer.release()
    video_fname = os.path.join(output_path, test_name+'.mp4')
    cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -qscale 0 {1}'.format(
       tmp_video_file.name, video_fname)).split()
    call(cmd)

    print("Adding audio")

    cmd = ('ffmpeg' + ' -i {0} -i {1}  -channel_layout stereo -qscale 0 {2}'.format(
       file_name, video_fname, video_fname.replace('.mp4', '_audio.mp4'))).split()
    call(cmd)

    if os.path.exists(video_fname):
        os.remove(video_fname)

if __name__ == '__main__':
    main()
