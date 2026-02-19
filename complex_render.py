import mitsuba as mi

# Set variant
mi.set_variant('scalar_rgb')

# Define material dictionaries for reuse
white_diffuse = {
    'type': 'diffuse',
    'reflectance': { 'type': 'rgb', 'value': [0.885809, 0.698859, 0.666422] }
}

red_diffuse = {
    'type': 'diffuse',
    'reflectance': { 'type': 'rgb', 'value': [0.570068, 0.0430135, 0.0443706] }
}

green_diffuse = {
    'type': 'diffuse',
    'reflectance': { 'type': 'rgb', 'value': [0.105421, 0.37798, 0.076425] }
}

glass_material = {
    'type': 'dielectric',
    'int_ior': 1.5,
    'ext_ior': 1.0,
}

gold_material = {
    'type': 'roughconductor',
    'material': 'Au',
    'distribution': 'ggx',
    'alpha': 0.05,
}

# Scene definition
scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'path',
        'max_depth': 8,
    },
    'sensor': {
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, 2, 10], # Backed up a bit
            target=[0, 2, 0],
            up=[0, 1, 0]
        ),
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 512,
            'pixel_format': 'rgba',
        },
        'sampler': {
            'type': 'independent',
            'sample_count': 256,
        },
    },
    
    # --- Geometry ---
    
    # Floor
    'floor': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.scale([2.5, 1, 2.5]).rotate(axis=[1, 0, 0], angle=-90),
        'bsdf': white_diffuse
    },
    
    # Ceiling
    'ceiling': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.translate([0, 4, 0]).scale([2.5, 1, 2.5]).rotate(axis=[1, 0, 0], angle=90),
        'bsdf': white_diffuse
    },
    
    # Back Wall
    'back': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.translate([0, 2, -2.5]).scale([2.5, 2, 1]),
        'bsdf': white_diffuse
    },
    
    # Left Wall (Red)
    'left': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.translate([-2.5, 2, 0]).scale([1, 2, 2.5]).rotate(axis=[0, 1, 0], angle=90),
        'bsdf': red_diffuse
    },
    
    # Right Wall (Green)
    'right': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.translate([2.5, 2, 0]).scale([1, 2, 2.5]).rotate(axis=[0, 1, 0], angle=-90),
        'bsdf': green_diffuse
    },
    
    # Light (Area Emitter)
    'light': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.translate([0, 3.99, 0]).scale([0.5, 1, 0.5]).rotate(axis=[1, 0, 0], angle=90),
        'emitter': {
            'type': 'area',
            'radiance': {
                'type': 'rgb',
                'value': [20, 20, 20],
            }
        },
        'bsdf': {'type': 'null'} # Invisible emitter surface
    },
    
    # Glass Sphere (Left)
    'sphere_glass': {
        'type': 'sphere',
        'center': [-1.0, 0.8, -0.5],
        'radius': 0.8,
        'bsdf': glass_material
    },
    
    # Gold Sphere (Right)
    'sphere_gold': {
        'type': 'sphere',
        'center': [1.0, 0.8, 0.5],
        'radius': 0.8,
        'bsdf': gold_material
    }
}

print("Loading scene...")
scene = mi.load_dict(scene_dict)

print("Rendering...")
image = mi.render(scene)

print("Saving complex_output.png")
mi.util.write_bitmap('complex_output.png', image)
