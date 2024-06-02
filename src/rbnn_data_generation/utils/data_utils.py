import numpy as np, matplotlib.pyplot as plt
import sys, bpy, os
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation as R

class ImageGeneratorBatch:
    def __init__(self, R, object, savepath, filepath, size, ratio, quality):
        super(ImageGeneratorBatch, self).__init__()
        self.R = R
        self.object = object
        self.savepath = savepath
        self.filepath = filepath
        self.size = size
        self.ratio = ratio
        self.quality = quality

    def generate_image(self):
        # Deleting default primitive cube
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()
        bpy.ops.wm.open_mainfile(filepath=self.filepath)

        # Setting up rending scaling and size
        size = self.size
        ratio = self.ratio
        bpy.context.scene.render.resolution_x = size*ratio
        bpy.context.scene.render.resolution_y = size*ratio
        bpy.context.scene.render.resolution_percentage = int(100/ratio)

        # Adding freely rotating rigid body and updating camera
        origin = (0, 0, 0)
        b_empty = bpy.data.objects.new("Empty", None)
        b_empty.location = origin
        b_empty = b_empty
        cam = bpy.context.scene.objects['Camera']
        cam.location = (0, 6, 0)
        
        cam_constraint = cam.constraints.new(type='TRACK_TO')
        b_empty = b_empty
        cam_constraint.target = b_empty
        cam.parent = b_empty
        bpy.context.collection.objects.link(b_empty)
        bpy.context.view_layer.objects.active = b_empty

        #import pdb; pdb.set_trace()

        obj = bpy.data.objects[self.object]
        obj.rotation_mode = 'QUATERNION'

        # Loading trajectory and rendering rest of the scene
        quat = R.from_matrix(self.R.view(-1, 3, 3).cpu().detach().numpy()).as_quat(canonical=True)
        quat_ = quat.reshape(self.R.shape[0], self.R.shape[1], 4)

        for n_batch in range(self.R.shape[0]):
            batch_dir = self.savepath + f'/traj-{n_batch}/' 
            os.makedirs(batch_dir, exist_ok=True)

            for t_step in range(self.R.shape[1]):
                quaternion = quat_[n_batch, t_step, ...] # rc.matrix_to_quaternion(self.R[n_batch, t_step, ...]).cpu().detach().numpy()
                obj.rotation_quaternion = tuple(float(x) for x in quaternion) # b_empty
                bpy.context.scene.render.image_settings.quality = self.quality
                filename = batch_dir + f'step-{t_step:06}.png'
                bpy.context.scene.render.filepath = filename
                bpy.ops.render.render(write_still=True)