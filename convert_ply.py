
import sys
import numpy as np
from plyfile import PlyData, PlyElement

def prune_spherical_harmonics(f):
    plydata = PlyData.read(f)
    retained_props = list()
    assert(plydata.elements[0].name == 'vertex')
    # TODO: verify there's no other element?
    for prop in plydata.elements[0].properties:
        if prop.name.startswith('f_rest'):
            continue
        retained_props.append(prop)

    # Create target dtype, based on retained properties.
    retained_dtype = list()
    for prop in retained_props:
        # Need to remove the '=' prefix.
        retained_dtype.append((prop.name, prop.dtype()[1:]))
    print(retained_dtype)
    print(len(retained_props))

    # Create target data.
    props_data = list()
    for prop in retained_props:
        props_data.append(plydata['vertex'][prop.name])

    # Convert to structured array.
    target_vertices = np.core.records.fromarrays(np.vstack(props_data), dtype=retained_dtype)
    
    print(target_vertices.shape)
    print(target_vertices.dtype)

    #dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    #plydata.elements[0].properties
    #plydata['vertex']['x']

    el = PlyElement.describe(target_vertices, 'vertex')
    PlyData([el]).write('output.ply')

if __name__ == "__main__":
    with open(sys.argv[1], 'rb') as f:
        prune_spherical_harmonics(f)
        