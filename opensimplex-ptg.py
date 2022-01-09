import numpy as np
from opensimplex import OpenSimplex
import plotly.graph_objects as go
import argparse, torch, time, random

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

FEATURE_SIZE = 64

def noise_slices(size: int, device: str, slice: str)-> None:
    
    #Setting up the generator and OpenSimplex class
    generator = torch.Generator(device = device)
    os = OpenSimplex(generator = generator, seed = random.randint(0, 65536))

    # Lazy size/zoom argument change - mostly because it's annoying
    # to do this over and over every time you type down size*64
    size = size*256

    # Creating dataset [-1,1] in slices, depending on the argument given
    # For some reason, these numbers tend to overflow at set number pairs,
    # i.e (0,0), (3,3), (4,2), (2,4), and many more which I don't understand why
    # then again, it's probably just a poor implementation of the noise generator
    # or my machine is doing the cha cha slide a bit too hard
    if slice == 'xy':
        z = [[int((os.noise3(i/FEATURE_SIZE, j/FEATURE_SIZE, 0.0)+1)*size*2) for i in range(1, size+1)] for j in range(1, size+1)]
    elif slice == 'yz':
        z = [[int((os.noise3(0.0, i/FEATURE_SIZE, j/FEATURE_SIZE)+1)*size*2) for i in range(1, size+1)] for j in range(1, size+1)]
    elif slice == 'xz':
        z = [[int((os.noise3(i/FEATURE_SIZE, 0.0, j/FEATURE_SIZE)+1)*size*2) for i in range(1, size+1)] for j in range(1, size+1)]

    elif slice == '2d':
        z = [[int((os.noise2(i/FEATURE_SIZE, j/FEATURE_SIZE)+1)*size*2) for i in range(1, size)] for j in range(size)]

    return z

# ------------------------- TODO: Make a separate function for generating a 3D shape out of 3 1D slices for the hell of it 

def create_graph(values):
    # General-use color scale (deep ocean -> snow)
    colorscale = [[0, 'rgb(0, 16, 84)'],
                [0.1, 'rgb(0,59,97)'],
                [0.2, 'rgb(4,105,151)'],
                [0.3, 'rgb(49,141,178)'],
                [0.4, 'rgb(41,164,195)'],
                [0.5, 'rgb(48,177,206)'],
                [0.6, 'rgb(180,170,141)'],
                [0.7, 'rgb(159,98,16)'],
                [0.8, 'rgb(86,46,25)'],
                [0.9, 'rgb(34,139,34)'],
                [1, 'rgb(255,255,255)']]

    z = values
    sh1, sh2 = len(z), len(z[0])
    x, y = np.linspace(0, 1, sh1), np.linspace(0, 1, sh2)

    # Creating figure & interactible plot
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
    fig.update_layout(title='Randomly Generated Terrain using (open)Simplex noise', autosize = False, height = 1024, width = 1024, margin=dict(l=65, r=50, b=65, t=90))
    fig.update_traces(colorscale=colorscale)

    # Show plot
    fig.show()
    # Saving doesn't work for whatever reason, I couldn't care less man it's 4am
    # fig.write_image(f'images/plot_3d_noise({np.datetime_as_string})')

def parse_args():
    # TODO: Implement these some time later because I need to finish my project ASAP
    parser = argparse.ArgumentParser(description="Select where the dataset and subsequent image will be generated through (Parallel, or CUDA)")
    generator = parser.add_mutually_exclusive_group()
    generator.add_argument(
        '-p', '--cpu',
        action='store_true',
        help='Generate plot using all CPU cores'
    )
    generator.add_argument(
        '-d', '--cuda',
        action='store_true',
        help='Generate plot using CUDA cores'
    )
    parser.add_argument(
        '-z', '--zoom',
        nargs='?',
        const='arg_was_not_given',
        type=int,
        default=1,
        help='Zoom in which the program creates data (Default 1x128)'
    )
    return parser.parse_args()

if __name__ == '__main__':
    # These will have to do for now, 
    # adding the argument parser will be also a pain in my ---
    driver = 'cpu'
    slice = 'xy'
    start = time.time()
    create_graph(noise_slices(1, driver, slice))
    end = time.time()
    print(f'Time taken to create and show 3D graph: {end-start} on device: {driver}')
    slice = '2d'
    start = time.time()
    create_graph(noise_slices(1, driver, slice))
    end = time.time()
    print(f'Time taken to create and show 2D graph: {end-start} on device: {driver}')

    driver = 'cuda'
    slice = 'xy'
    start = time.time()
    create_graph(noise_slices(1, driver, slice))
    end = time.time()
    print(f'Time taken to create and show 3D graph: {end-start} on device: {driver}')
    slice = '2d'
    start = time.time()
    create_graph(noise_slices(1, driver, slice))
    end = time.time()
    print(f'Time taken to create and show 2D graph: {end-start} on device: {driver}')