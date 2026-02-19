import mitsuba as mi

# Set the variant to scalar_rgb
mi.set_variant('scalar_rgb')

# Create a basic scene
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': 'path',
    },
    'sensor': {
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, 0, 5],
            target=[0, 0, 0],
            up=[0, 1, 0]
        ),
        'film': {
            'type': 'hdrfilm',
            'width': 256,
            'height': 256,
        },
    },
    'light': {
        'type': 'point',
        'position': [5, 5, 5],
        'intensity': {
            'type': 'rgb',
            'value': [100, 100, 100],
        },
    },
    'shape': {
        'type': 'sphere',
        'center': [0, 0, 0],
        'radius': 1.0,
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.2, 0.7, 0.2],
            }
        },
    },
})

# Render the scene
image = mi.render(scene, spp=128)

# Write the output image
mi.util.write_bitmap('output.png', image)
print("Rendered image saved to output.png")