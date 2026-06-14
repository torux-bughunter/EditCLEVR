from __future__ import print_function
import argparse, json, math, os, sys

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
parser.add_argument('--properties_json', default='data/properties.json')
parser.add_argument('--shape_dir', default='data/shapes')
parser.add_argument('--material_dir', default='data/materials')
parser.add_argument('--object_index', required=True, type=int)
parser.add_argument('--factor', required=True, choices=['color', 'material', 'size', 'shape'])
parser.add_argument('--new_value', required=True)
parser.add_argument('--output_blendfile', default=None)


def _load_properties(path):
  with open(path, 'r') as f:
    props = json.load(f)
  color_map = {name: [float(c) / 255.0 for c in rgb] + [1.0] for name, rgb in props['colors'].items()}
  return props, color_map


def _set_active_object(obj):
  bpy.ops.object.select_all(action='DESELECT')
  obj.select_set(True)
  bpy.context.view_layer.objects.active = obj


def _replace_material(obj, material_name, rgba):
  _set_active_object(obj)
  while len(obj.data.materials) > 0:
    obj.data.materials.pop(index=0)
  utils.add_material(material_name, Color=rgba)


def _replace_shape(obj, output_shape_name, size_name, shape_dir, props):
  x, y, _ = obj.location
  theta = obj.rotation_euler[2]
  human_to_blend = props['shapes']
  internal_shape = human_to_blend[output_shape_name]
  scale = props['sizes'][size_name]
  if internal_shape == 'SmoothCube_v2':
    scale /= math.sqrt(2)
  utils.delete_object(obj)
  utils.add_object(shape_dir, internal_shape, scale, (x, y), theta=float(theta))
  return bpy.context.object


def _rebuild_object_record(obj, original, camera):
  pixel_coords = utils.get_camera_coords(camera, obj.location)
  return {
    'blender_name': obj.name,
    'shape': original['shape'],
    'size': original['size'],
    'material': original['material'],
    '3d_coords': tuple(obj.location),
    'rotation': float(obj.rotation_euler[2]),
    'pixel_coords': pixel_coords,
    'color': original['color'],
  }


def main(args):
  bpy.ops.wm.open_mainfile(filepath=args.input_blendfile)
  utils.load_materials(args.material_dir)
  for path in [args.output_image, args.output_scene, args.output_mask, args.output_blendfile]:
    if path:
      os.makedirs(os.path.dirname(path), exist_ok=True)

  with open(args.input_scene_json, 'r') as f:
    scene_struct = json.load(f)
  scene_struct['image_filename'] = os.path.basename(args.output_image)
  scene_struct['mask_filename'] = os.path.basename(args.output_mask)

  props, color_map = _load_properties(args.properties_json)
  objects = scene_struct['objects']
  target_record = dict(objects[args.object_index])
  target_obj = bpy.data.objects[target_record['blender_name']]
  if target_record[args.factor] == args.new_value:
    raise ValueError('Atomic edit must change %s from %s to a different value' % (args.factor, args.new_value))

  if args.factor == 'color':
    target_record['color'] = args.new_value
  elif args.factor == 'material':
    target_record['material'] = args.new_value
  elif args.factor == 'size':
    size_ratio = props['sizes'][args.new_value] / props['sizes'][target_record['size']]
    target_obj.scale = [value * size_ratio for value in target_obj.scale]
    target_obj.location[2] = props['sizes'][args.new_value]
    target_record['size'] = args.new_value
  elif args.factor == 'shape':
    target_record['shape'] = args.new_value
    target_obj = _replace_shape(
      target_obj,
      output_shape_name=args.new_value,
      size_name=target_record['size'],
      shape_dir=args.shape_dir,
      props=props,
    )

  rgba = color_map[target_record['color']]
  material_name = props['materials'][target_record['material']]
  _replace_material(target_obj, material_name, rgba)

  camera = render_images.get_object_by_names('Camera')
  updated_objects = []
  for idx, record in enumerate(objects):
    current_record = dict(record)
    if idx == args.object_index:
      current_record.update(target_record)
      current_record['blender_name'] = target_obj.name
      obj = target_obj
    else:
      obj = bpy.data.objects[current_record['blender_name']]
    updated_objects.append(_rebuild_object_record(obj, current_record, camera))

  scene_struct['objects'] = updated_objects
  scene_struct['relationships'] = render_images.compute_all_relationships(scene_struct)

  scene = bpy.context.scene
  scene.render.filepath = args.output_image
  bpy.ops.render.render(write_still=True)
  render_images.render_mask_image([bpy.data.objects[obj['blender_name']] for obj in updated_objects], path=args.output_mask)

  with open(args.output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)
  if args.output_blendfile:
    bpy.ops.wm.save_as_mainfile(filepath=args.output_blendfile)


if __name__ == '__main__':
  if INSIDE_BLENDER:
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
