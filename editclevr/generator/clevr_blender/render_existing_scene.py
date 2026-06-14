from __future__ import print_function
import argparse, json, os, random, sys

INSIDE_BLENDER = True
try:
  import bpy
except ImportError:
  INSIDE_BLENDER = False

if INSIDE_BLENDER:
  script_dir = os.path.dirname(os.path.abspath(__file__))
  if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
  import utils
  import render_images


parser = argparse.ArgumentParser()
parser.add_argument('--input_blendfile', required=True)
parser.add_argument('--input_scene_json', required=True)
parser.add_argument('--output_image', required=True)
parser.add_argument('--output_scene', required=True)
parser.add_argument('--output_mask', required=True)
parser.add_argument('--material_dir', default='data/materials')
parser.add_argument('--camera_jitter', type=float, default=0.15)
parser.add_argument('--light_jitter', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=0)


def _rebuild_object_record(obj, original, camera):
  pixel_coords = utils.get_camera_coords(camera, obj.location)
  current = dict(original)
  current['pixel_coords'] = pixel_coords
  return current


def main(args):
  for path in [args.output_image, args.output_scene, args.output_mask]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

  bpy.ops.wm.open_mainfile(filepath=args.input_blendfile)
  utils.load_materials(args.material_dir)

  with open(args.input_scene_json, 'r') as f:
    scene_struct = json.load(f)
  scene_struct['image_filename'] = os.path.basename(args.output_image)
  scene_struct['mask_filename'] = os.path.basename(args.output_mask)

  rng = random.Random(args.seed)
  camera = render_images.get_object_by_names('Camera')
  for i in range(3):
    camera.location[i] += args.camera_jitter * (rng.random() - 0.5)

  for light_name in [('Lamp_Key', 'Key_Light', 'Light_Key'),
                     ('Lamp_Fill', 'Fill_Light', 'Light_Fill'),
                     ('Lamp_Back', 'Back_Light', 'Light_Back')]:
    light = render_images.get_object_by_names(*light_name)
    for i in range(3):
      light.location[i] += args.light_jitter * (rng.random() - 0.5)

  updated_objects = []
  blender_objects = []
  for record in scene_struct['objects']:
    obj = bpy.data.objects[record['blender_name']]
    blender_objects.append(obj)
    updated_objects.append(_rebuild_object_record(obj, record, camera))
  scene_struct['objects'] = updated_objects
  scene_struct['relationships'] = render_images.compute_all_relationships(scene_struct)

  scene = bpy.context.scene
  scene.render.filepath = args.output_image
  bpy.ops.render.render(write_still=True)
  render_images.render_mask_image(blender_objects, path=args.output_mask)

  with open(args.output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)


if __name__ == '__main__':
  if INSIDE_BLENDER:
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
