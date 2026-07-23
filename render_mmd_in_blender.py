"""Import a PMX model and VMD motion with mmd-tools, then render a video."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_arguments():
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmx", type=Path, required=True)
    parser.add_argument("--vmd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args(arguments)


def look_at(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def evaluated_bounds(objects, frame):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    points = []
    for obj in objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        points.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
    minimum = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maximum = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return minimum, maximum


def add_area_light(name, location, energy, size, color):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    look_at(light, (0, 0, 1))
    return light


def build_scene(arguments):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.preferences.addon_enable(module="mmd_tools")

    bpy.ops.mmd_tools.import_model(
        filepath=str(arguments.pmx),
        types={"MESH", "ARMATURE", "DISPLAY", "MORPHS"},
        scale=0.08,
        clean_model=True,
        rename_bones=False,
        log_level="WARNING",
    )
    imported_objects = list(bpy.context.scene.objects)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = next(obj for obj in imported_objects if obj.type == "ARMATURE")
    bpy.ops.mmd_tools.import_vmd(
        directory=str(arguments.vmd.parent) + os.sep,
        files=[{"name": arguments.vmd.name}],
        scale=0.08,
        margin=0,
        bone_mapper="PMX",
        update_scene_settings=True,
        use_NLA=False,
    )
    action_count = len(bpy.data.actions)
    keyframe_count = sum(
        len(curve.keyframe_points)
        for action in bpy.data.actions
        for curve in action.fcurves
    )
    if action_count == 0 or keyframe_count == 0:
        raise RuntimeError("mmd-tools did not bind any VMD keyframes to the imported model")

    scene = bpy.context.scene
    scene.frame_start = 0
    # The source dance clip is 300 frames: keep the rendered range exactly
    # 0..299 even if mmd-tools imports a trailing VMD keyframe.
    scene.frame_end = 299
    scene.render.fps = 30
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 32
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(arguments.output)
    scene.render.film_transparent = False
    scene.view_settings.look = "Medium High Contrast"

    world = bpy.data.worlds.new("Studio World") if not scene.world else scene.world
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.018, 0.024, 0.035, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.28

    middle_frame = (scene.frame_start + scene.frame_end) // 2
    minimum, maximum = evaluated_bounds(imported_objects, middle_frame)
    center = (minimum + maximum) / 2
    height = max(maximum.z - minimum.z, 1.0)
    width = max(maximum.x - minimum.x, maximum.y - minimum.y, 1.0)

    camera_data = bpy.data.cameras.new("Dance Camera")
    camera = bpy.data.objects.new("Dance Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = 52
    camera_data.dof.use_dof = False
    distance = max(height * 2.85, width * 2.6)
    camera.location = (center.x, center.y - distance, minimum.z + height * 0.56)
    look_at(camera, (center.x, center.y, minimum.z + height * 0.48))

    key = add_area_light(
        "Key Light", (center.x - height * 0.8, center.y - height * 1.0, minimum.z + height * 1.6),
        1100, height * 0.9, (1.0, 0.82, 0.72))
    look_at(key, center)
    fill = add_area_light(
        "Fill Light", (center.x + height, center.y - height * 0.3, minimum.z + height),
        850, height, (0.52, 0.72, 1.0))
    look_at(fill, center)
    rim = add_area_light(
        "Rim Light", (center.x, center.y + height * 0.8, minimum.z + height * 1.25),
        1250, height * 0.7, (0.65, 0.8, 1.0))
    look_at(rim, center)

    scene.frame_set(0)
    blend_path = arguments.output.with_suffix(".blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    return scene, middle_frame


def main():
    arguments = parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    scene, middle_frame = build_scene(arguments)
    if arguments.preview:
        scene.frame_set(middle_frame)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(arguments.output.with_suffix(".png"))
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
