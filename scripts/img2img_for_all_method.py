import modules.images as images
import modules.scripts as scripts
import modules.sd_samplers as sd_samplers
import gradio as gr
import os
import glob
import random
import operator

from modules.processing import process_images, StableDiffusionProcessingImg2Img, Processed
import cv2
import copy
import numpy as np
from PIL import Image
import time
import json

def resize_img(img, w, h):
    if img.shape[0] + img.shape[1] < h + w:
        interpolation = interpolation=cv2.INTER_CUBIC
    else:
        interpolation = interpolation=cv2.INTER_AREA

    return cv2.resize(img, (w, h), interpolation=interpolation)



def draw_grid(rows, image_list, x_labels, y_labels):
    ver_texts = [[images.GridAnnotation(y)] for y in y_labels]
    hor_texts = [[images.GridAnnotation(x)] for x in x_labels]

    w = image_list[0].width
    h = image_list[0].height

    grid = images.image_grid(image_list, rows=rows)
    grid = images.draw_grid_annotations(grid, w, h, hor_texts, ver_texts)

    return grid


class Script(scripts.Script):

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "img2img for all method"


# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):
        return is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        input_dir = gr.Textbox(label='Input directory', lines=1)
        mask_dir = gr.Textbox(label='Mask directory(optional)', lines=1)
        output_dir = gr.Textbox(label='Output directory', lines=1)

        chs = ["All"]
        chs.extend([x.name for x in sd_samplers.samplers_for_img2img])
        sampler = gr.Dropdown(label='Sampling method', choices=chs, value=chs[1], type="value")
        step_and_cfg_list = gr.Textbox(label='[Sampling Steps, CFG Scale] List', lines=1, value="[20,7],[50,20]")
#        dstr_list = gr.Textbox(label='Denoising Strength List', lines=1, value="0.1,0.15,0.2,0.25,0.3,0.35")
        dstr_list = gr.Textbox(label='Denoising Strength List', lines=1, value="0.1,0.35")
        repeat_count = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Img2Img Repeat Count")
        inc_seed = gr.Slider(minimum=0, maximum=9999999, step=1, value=0, label="Add N to seed when repeating")

        return [input_dir, mask_dir, output_dir, sampler, step_and_cfg_list, dstr_list, repeat_count, inc_seed]
  

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p:StableDiffusionProcessingImg2Img, input_dir, mask_dir, output_dir, sampler, step_and_cfg_list_str, dstr_list_str, img2img_repeat_count, inc_seed):
        args = locals()
        result = []

        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        print(step_and_cfg_list_str)
        print(dstr_list_str)

        step_and_cfg_list = json.loads("[" + step_and_cfg_list_str + "]")
        dstr_list = json.loads("[" + dstr_list_str + "]")

        print(step_and_cfg_list)
        print(dstr_list)
        

        def get_img_filename(p:StableDiffusionProcessingImg2Img, basename, i):
            def get_step_str(steps):
                return "Step"+str(steps)
            def get_cfg_str(cfg):
                return "Cfg"+str(cfg)
            def get_dstr_str(dstr):
                return "Dstr"+str(dstr)
            def get_repeat_str(repeat):
                return "Repeat"+str(repeat+1)

            return basename + "_" + p.sampler_name +\
                    "_" + get_step_str(p.steps) +\
                    "_" + get_cfg_str(p.cfg_scale) +\
                    "_" + get_dstr_str(p.denoising_strength) +\
                    "_" + get_repeat_str(i) + ".png"
        
        def append_result(p:StableDiffusionProcessingImg2Img, i, imgname, saved_path, seed):
            item = {
                "base_img_name" : imgname,
                "sampler_name" : p.sampler_name,
                "steps" : p.steps,
                "cfg_scale" : p.cfg_scale,
                "denoising_strength" : p.denoising_strength,
                "repeat_count" : i + 1,
                "saved_path" : saved_path,
                "seed" : seed
            }
            result.append(item)

        def img2img_process(p:StableDiffusionProcessingImg2Img, i, img_path):
            _p = copy.copy(p)

            proc = process_images( _p )
            print(proc.seed)

            base_name = os.path.splitext(os.path.basename(img_path))[0]

            filename = get_img_filename( p, base_name, i )
            save_path = os.path.join( output_dir , filename )
            proc.images[0].save( save_path )

            append_result(p, i, base_name, save_path, proc.seed)

            return proc

        def img2img_for_denoising_strength(p:StableDiffusionProcessingImg2Img, img_path):
            mask = p.image_mask
            resized_mask = None

            for i in range(img2img_repeat_count):
                proc = img2img_process(p, i, img_path)
                p.init_images=[proc.images[0]]
                if mask is not None and resized_mask is None:
                    resized_mask = resize_img(np.array(mask) , proc.images[0].width, proc.images[0].height)
                    resized_mask = Image.fromarray(resized_mask)
                p.image_mask = resized_mask
                p.seed = (proc.seed + inc_seed)

            return proc

        def img2img_for_cfg_and_steps(p:StableDiffusionProcessingImg2Img, img_path):
            for ds in dstr_list:
                _p = copy.copy(p)
                _p.denoising_strength = ds
                proc = img2img_for_denoising_strength(_p, img_path)
            return proc
        
        def img2img_for_method(p:StableDiffusionProcessingImg2Img, img_path):
            for steps,cfg in step_and_cfg_list:
                _p = copy.copy(p)
                _p.steps = steps
                _p.cfg_scale = cfg
                proc = img2img_for_cfg_and_steps(_p, img_path)
            return proc

        

        ########### start

        if not os.path.isdir(input_dir):
            print("input_dir not found")
            return Processed()

        if not os.path.isdir(output_dir):
            print("output_dir not found")
            return Processed()
        
        if step_and_cfg_list == None or len(step_and_cfg_list) == 0:
            print("(Sampling Steps, CFG Scale) List cannot be parsed.")
            return Processed()

        if dstr_list == None or len(dstr_list) == 0:
            print("Denoising Strength List cannot be parsed.")
            return Processed()
        
        for s,c in step_and_cfg_list:
            if s == None or c == None:
                print("(Sampling Steps, CFG Scale) List cannot be parsed.2")
                return Processed()

        
        root_output_dir = output_dir = os.path.join(output_dir, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(output_dir, exist_ok=False)
        output_dir = os.path.join(output_dir, "imgs")
        os.makedirs(output_dir, exist_ok=False)

        p.do_not_save_samples = True
        p.do_not_save_grid = True

        imgs = glob.glob( os.path.join(input_dir ,"*.png") )
        for img in imgs:

            image = Image.open(img)
            mask = None

            img_basename = os.path.basename(img)
            mask_path = os.path.join( mask_dir , img_basename )

            if os.path.isfile( mask_path ):
                mask = Image.open(mask_path)

            _p = copy.copy(p)
            _p.init_images=[image]
            _p.image_mask = mask

            if sampler == "All":
                for method in sd_samplers.samplers_for_img2img:
                    _p.sampler_name = method.name
                    proc = img2img_for_method(_p, img)
            else:
                _p.sampler_name = sampler
                proc = img2img_for_method(_p, img)

        ### param txt
        with open( os.path.join( root_output_dir ,"param.txt" ), "w") as f:
            f.write(proc.info)
        with open( os.path.join( root_output_dir ,"args.txt" ), "w") as f:
            f.write(str(args))
        with open( os.path.join( root_output_dir ,"output.txt" ), "w") as f:
            f.write(json.dumps(result, indent=4))




        ### create grid
        def create_grid_per_img(method_name, img_path):
            print("create grid per image")
            print(method_name)

            image_list = []
            image_name = os.path.splitext(os.path.basename(img_path))[0]

            items = list(filter(lambda d: d["base_img_name"] == image_name and d["sampler_name"] == method_name, result))
            items = sorted(items, key=operator.itemgetter("steps","cfg_scale","denoising_strength","repeat_count"))

            for item in items:
                image_list.append( Image.open(item["saved_path"]) )

            label_x = [ "repeat " + str(x+1) for x in range(img2img_repeat_count) ]

            items = list(filter(lambda d: d["repeat_count"] == 1, items))

            label_y = [ "steps :" + str(y["steps"]) +", cfg_scale :"+ str(y["cfg_scale"]) +", denoising_strength :"+ str(y["denoising_strength"]) for y in items]

            grid = draw_grid( len(label_y), image_list, label_x, label_y)

            save_path = os.path.join(root_output_dir, image_name + "_" + method_name + ".png")

            grid.save( save_path )

            return grid

        def create_grid(method_name, img_path_list):
            print("create grid")
            print(method_name)
            image_list = []

            tmp = Image.open(result[0]["saved_path"])
            width = tmp.width
            height = tmp.height

            items = list(filter(lambda d: d["sampler_name"] == method_name, result))
            items = sorted(items, key=operator.itemgetter("steps","cfg_scale","denoising_strength","repeat_count"))

            for img in img_path_list:
                image_list.append( Image.open(img).resize((width, height), Image.LANCZOS) )

            for item in items:
                image_list.append( Image.open(item["saved_path"]) )

            label_x = [ os.path.basename(img) for img in img_path_list ]

            image_name = os.path.splitext(os.path.basename(img_path_list[0]))[0]
            items = list(filter(lambda d: d["base_img_name"] == image_name, items))

            label_y = ["original"]
            label_y.extend([ "steps :" + str(y["steps"]) +", cfg_scale :"+ str(y["cfg_scale"]) +", denoising_strength :"+ str(y["denoising_strength"]) +", repeat :" + str(y["repeat_count"]) for y in items])

            grid = draw_grid( len(label_y), image_list, label_x, label_y)

            save_path = os.path.join(root_output_dir, method_name + ".png")

            grid.save( save_path )

            return grid

        def create_grid_all(img_path):
            print("create grid all")
            image_list = []

            tmp = Image.open(result[0]["saved_path"])
            width = tmp.width
            height = tmp.height

            image_name = os.path.splitext(os.path.basename(img_path))[0]

            items = list(filter(lambda d: d["base_img_name"] == image_name, result))
            items = sorted(items, key=operator.itemgetter("steps","cfg_scale","denoising_strength","repeat_count"))

            label_x = ["original"]
            label_x.extend([ method.name for method in sd_samplers.samplers_for_img2img ])

            label_items = list(filter(lambda d: d["sampler_name"] == "Euler a", items))
            label_y = ([ "steps :" + str(y["steps"]) +", cfg_scale :"+ str(y["cfg_scale"]) +", denoising_strength :"+ str(y["denoising_strength"]) +", repeat :" + str(y["repeat_count"]) for y in label_items])

            org_image = Image.open(img_path).resize((width, height), Image.LANCZOS)

            for index_item in label_items:
                for method in label_x:
                    if method == "original":
                        image_list.append( org_image )
                    else:
                        item = list(filter(lambda d: d["steps"] == index_item["steps"] and\
                                                d["cfg_scale"] == index_item["cfg_scale"] and\
                                                d["denoising_strength"] == index_item["denoising_strength"] and\
                                                d["repeat_count"] == index_item["repeat_count"] and\
                                                d["sampler_name"] == method \
                                                , items))[0]
                        image_list.append( Image.open(item["saved_path"]) )


            grid = draw_grid( len(label_y), image_list, label_x, label_y)

            save_path = os.path.join(root_output_dir, image_name + "_ALL.png")

            grid.save( save_path )

            grid = grid.resize((grid.width//3, grid.height//3), Image.LANCZOS)
            save_path = os.path.join(root_output_dir, image_name + "_small_ALL.png")
            grid.save( save_path )

            return grid


        # x ... repeat
        # y ... (step + cfg)*str

        for img in imgs:
            if sampler == "All":
                for method in sd_samplers.samplers_for_img2img:
                    grid = create_grid_per_img( method.name, img )
            else:
                grid = create_grid_per_img( sampler, img )


        if len(imgs) > 1:
            if sampler == "All":
                for method in sd_samplers.samplers_for_img2img:
                    grid = create_grid( method.name, imgs )
            else:
                grid = create_grid( sampler, imgs )
        
        if sampler == "All":
            grid = create_grid_all( imgs[0] )

#        proc.images[0] = grid

        return proc
