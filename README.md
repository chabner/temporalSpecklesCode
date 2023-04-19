# Spatio-temporal Monte Carlo speckle statistics and field simulator

Spatio-temporal Monte Carlo speckle statistics and field simulator is a next-event estimation path tracing simulation for rendering spatio-temporal speckle statistics and fields, accelerated on GPU and written in CUDA.

## Table of context

1. [ Prerequisites. ](#prereq)
2. [ Installation. ](#inst)
3. [ Prerequisites. ](#prereq)
4. [ Usage. ](#usage)
5. [Code configuration.](#config)
    1. [Simulator.](#sim)
    2. [Medium.](#med)
    3. [Scattering.](#scat)
    4. [Source.](#source)
    5. [Sampler.](#sampler)
    6. [Tracer.](#tracer)
6. [Program execution.](#exe)
7. [References.](#ref)

<a name="prereq"></a>
## Prerequisites

The package requires MATLAB R2022a or later and CUDA toolkit 11.6 or later. You can download MATLAB from [MathWorks](https://www.mathworks.com/products/matlab.html), and CUDA toolkit from [NVIDIA](https://developer.nvidia.com/cuda-toolkit). In addition, cuBLAS library is required to compile the code. This library is a part of the CUDA toolkit, but you may need to add its library to MATLAB compiler path. Please refer to the "-L" option in [MATLAB mex](https://www.mathworks.com/help/matlab/ref/mex.html#btw17rw-1-option1optionN) compiler.

<a name="inst"></a>
## Installation

To install the package, follow these steps:

1. Clone or download the repository to your local machine.

2. Open MATLAB, navigate to the cloned or downloaded repository folder, and run
```matlab
buildMex.m
```
This will compile the required code files.

3. Add the compiled code library to MATLAB path by running the following command:

```matlab
addpath(genpath('/path/to/temporalSpecklesCode'))
```

<a name="usage"></a>
## Usage

The compiled code contains two main functions for generating spatio-temporal speckle patterns:

1. sstmc3D: Use this function to generate speckle patterns in 3D scenes.

2. sstmc2D: Use this function to generate speckle patterns in 2D scenes.

To use either function, call it with the required input configuration. The input configuration is detailed in the function documentation.

In 3D scenes, we typically refer to x, y, and z coordinates, whereas in 2D scenes, we usually use x and y coordinates in a similar manner.

<a name="config"></a>
## Code configuration
In this section, we provide instructions on how to configure the code.

<a name="sim"></a>
### Simulator

To adjust the number of rendered concurrent paths, you can modify the following fields in the `config.simulation` structure:

```matlab
config.simulation.deviceNumber
```

This is an optional field that accepts an integer greater than 0. The default value is 0, which corresponds to the first GPU device in system. You can set this field to the device number of your choice.

```matlab
config.simulation.batchSize
```

This is an optional field that accepts an integer greater than 1. The default value is 1024. Increasing the `batchSize` usually results in better GPU utilization, but it is limited by the GPU memory.

```matlab
config.simulation.fullIteration
```

This is an optional field that accepts an integer greater than 1. The default value is 1, which corresponds to a single iteration with `batchSize` paths. You can set this field to increase the number of iterations. This value is not limited by the GPU memory.

```matlab
config.simulation.renderingsNum
```

This is an optional field that accepts an integer greater than 1. The default value is 1, which corresponds to a single simulation run. You can set this field to the number of simulations you want to run.

```matlab
config.simulation.lambda
```

This is an optional field that accepts a float type value. The default value is 1.0, which corresponds to the wavelength. You can adjust this value to simulate speckles with different wavelengths.

<a name="med"></a>
### Medium

In the following section, we will provide instructions on how to configurate `config.medium`, the medium in the simulator representing the physical environment through which the light propagates.

```matlab
config.medium.type = 'Heterogeneous';
```

This field is required and defines a heterogeneous volume. Homogenous volume can also be defined by using a heterogeneous volume with a single bin. The bins in the x, y, and, for 3D, z axis are defined as follows:

```matlab
config.medium.xAxis = [ ... ];
config.medium.yAxis = [ ... ];
config.medium.zAxis = [ ... ];
```

Each axis is a vector containing at least two increasing elements from the beginning of the volume until it ends for each axis. For a homogenous medium, each axis contains exactly two elements defining a volume with a single bin.

In a heterogeneous volume, the material number is defined in each bin, which specifies the volumetric properties of the volume in the corresponding bin. The number of bins in the volume is calculated as follows:

```matlab
(numel(config.medium.xAxis) - 1) * ...
(numel(config.medium.yAxis) - 1) * ...
(numel(config.medium.zAxis) - 1)
```

Material 0 is reserved for outside the volume, and materials 1 and above are defined in

```matlab
config.medium.materialGrid
```

The number of materials is defined as `M`, which can be calculated by:

```matlab
M = max(config.medium.materialGrid(:)) + 1;
```

The scattering coefficient is defined as

```matlab
config.medium.sigs = [0, ... ];
```
This field is required and defined as a vector with `M` elements. The first element is the scattering coefficient outside the volume, and should be defined as 0. The rest of the elements are the scattering coefficient for the different materials, starting from material 1. Each element is a positive floating number larger than 0.

```matlab
config.medium.siga
```

This field is optional, and the default value is a vector of `M` zeros. It defines the absorption coefficient for each material, requiring `M` elements, where each element is a positive floating number larger than 0.

<a name="scat"></a>
### Scattering
    
The scattering of light is defined by the type and amplitude function of the medium. In this section, we provide instructions on how to configure `config.scattering`, the scattering parameters.

```matlab
config.scattering.type
```

This is a required field and should be a cell vector with `M` elements, where each element defines the scattering type for the corresponding material. The possible types are:

- `'Isotropic'`: for isotropic scattering.
- `'HG'`: for Henyey-Greenstein scattering.
- `'Tabular'`: for tabulated values scattering.
- `'vMF'`: for von Mises-Fisher mixture scattring type.

```matlab
config.scattering.amplitudeFunction
```

This is also a required field and should be a cell vector with `M` elements, where each element defines the amplitude function of the scattering type for the corresponding material. The definition of the amplitude function depends on the scattering type:

#### Isotropic scattering
```matlab
config.scattering.amplitudeFunction{1} = 'Isotropic';
```

Isotropic scattering does not require any value, and the cell can be left empty.

#### Henyey-Greenstein scattering
```matlab
config.scattering.amplitudeFunction{1} = 'HG';
```

Henyey-Greenstein scattering requires a floating point value between -1 and 1, which defines the anisotropy parameter `g` of the Henyey-Greenstein phase function.

#### Tabulated scattering
```matlab
config.scattering.amplitudeFunction{1} = 'Tabular';
```

Tabulated scattering requires a vector of `N` complex numbers that defines the amplitude function. In 3D, the amplitude function is defined from 0 to pi, where 0 corresponds to forward scattering and pi corresponds to backscattering. In 2D, the amplitude function is defined from 0 to 2pi.

#### von Mises-Fisher mixture scattering
```matlab
config.scattering.amplitudeFunction{1} = 'vMF';
```

von Mises-Fisher mixture scattering requires a struct with the fields `mixtureMu`, `mixtureC`, and `mixtureAlpha`. Each field is a vector of floating numbers that defines the mixture of von Mises-Fisher functions. Examples of von Mises-Fisher mixtures can be found in `confocalRenderings`.

<a name="source"></a>
### Source
The source is a fundamental component of the scene in the simulator, as it represents the light source that illuminates the object being simulated and the camera which collecting the scattered light in the observed scene. The type and properties of the source, such as its position, direction, and aperture, can have a significant impact on the resulting speckle pattern or image. Therefore, accurately defining the source is critical for generating realistic and accurate simulations.

The source in the simulator can be defined as `illumination` and `view`. Additionally, for the speckle correlation algorithm, the source can also be defined as `illumination2` and `view2`, where the number of elements of `illumination` and `illumination2`, as well as `view` and `view2` are respectively equal to `L` and `V`.

For the speckle field rendering algorithm, `illumination` and `view` are required fields. For the speckle correlation algorithm, `illumination`, `illumination2`, `view` and `view2` are all required fields.

In the following section, we will give an example of how to define the source `illumination`. The sources `view`, `illumination2`, and `view2` are defined similarly.

```matlab
config.illumination.type
```

This is a required field that defines the source type. It should be a character array.

#### Point source
```matlab
config.illumination.type = 'PointSource'
```

This defines infinitesimal spatial points of the source, which spread or sense the light isotropically. The spatial point positions are defined with:

```matlab
config.illumination.location.x
config.illumination.location.y
config.illumination.location.z
```

Each coordinate is a vector with `L` floating values.

#### Directional source
```matlab
config.illumination.type = 'FarField'
```

This defines directional sources, which are defined with:

```matlab
config.illumination.direction.x
config.illumination.direction.y
config.illumination.direction.z
```

Each coordinate is a vector with `L` floating values where:

```matlab
abs(config.illumination.direction.x).^2 + ...
abs(config.illumination.direction.y).^2 + ...
abs(config.illumination.direction.z).^2
```

is a vector of `L` ones.

Transmissive mode is defined when:

```matlab
config.illumination.direction.x * config.view.direction.x + ...
config.illumination.direction.y * config.view.direction.y + ...
config.illumination.direction.z * config.view.direction.z > 0
```

Backscattering mode is defined when:

```matlab
config.illumination.direction.x * config.view.direction.x + ...
config.illumination.direction.y * config.view.direction.y + ...
config.illumination.direction.z * config.view.direction.z < 0
```

#### Focused source
```matlab
config.illumination.type = 'GaussianBeam'
```

This defines a focused source:

```matlab
config.illumination.focalPoint.x
config.illumination.focalPoint.y
config.illumination.focalPoint.z
```

This is the focal point, which is defined similarly to the location of the point source. Each coordinate is a vector with `L` floating values.

```matlab
config.illumination.focalDirection.x
config.illumination.focalDirection.y
config.illumination.focalDirection.z
```

This is the focal direction, which is defined similarly to the direction of the directional source. Each coordinate is a vector with `L` floating values.

```matlab
config.illumination.aperture
```

This is a required field that defines the standard deviation of the Gaussian-shape aperture.

See the example `speckleRenderings` for examples of different source types.

<a name="sampler"></a>
### Sampler
The sampler `config.sampler` defines the simulation algorithm type and parameters.

There are two options for defining the sampler:
- For the speckle field algorithm, set `config.sampler.type` to `'TemporalSampler'`.
- For the speckle correlation algorithm, set `config.sampler.type` to `'TemporalCorrelationSampler'`.

```matlab
config.sampler.t
```

is a vector of increasing floating-point values with size `T`, which defines the time in the simulation. 

```matlab
config.sampler.D
```

is a vector of `M` floating-point values, where each value defines the Brownian motion parameter for the corresponding material.

```matlab
config.sampler.U.x
config.sampler.U.y
config.sampler.U.z
```

Each coordinate is a vector of `M` floating-point values that defines the linear motion parameter for the corresponding material.

<a name="tracer"></a>
### Tracer

The role of the tracer `config.tracer` in the simulation is to simulate the propagation of light through the scene. It tracks the path of the photons as they interact with the materials in the scene, and calculates the intensity and other properties of the resulting light. 

```matlab
config.tracer.type = 'nee';
```

The `type` parameter is used to define the type of tracer to be used in the simulation. For now, only next-event estimation (NEE) is supported, which can be specified by setting the value to `'nee'`.

```matlab
config.tracer.seed
```

The `seed` parameter is an optional field of integer type. Its default value is the current clock. It can be used to define a seed for generating random numbers.

```matlab
config.tracer.isCBS
```

The `isCBS` parameter is an optional field of bool type, which defaults to `false`. Coherent back scattering (CBS) considers reverse paths, but it also consumes additional computation time. To activate CBS, set `isCBS` to `true`. You can refer to the `mudiffCBSValidation2D` example for additional information.

<a name="exe"></a>
## Program execution
Once the configuration is set up, the code is executed in the following manner. For a 2D configuration, run the following command:

```matlab
[u, iter] = sstmc2D(config);
```

For a 3D configuration, use the following command:

```matlab
[u, iter] = sstmc3D(config);
```

Here, `u` is a 4D matrix of complex floating values, with a size of `L x V x T x I`, where `L` and `V` represent the number of illumination and view sources, `T` is the number of temporal points, and `I` is the number of simulations as defined in `renderingsNum`.

The variable `iter` is a vector of size `I` which defines the number of paths used for each simulation. The number of paths used can be calculated as

```matlab
config.simulation.batchSize * config.simulation.fullIteration + missPaths
```

Where `missPaths` represents the number of paths that are missing the volume and contribute a value of 0 to the result. The result can be normalized according to `iter`, where the speckle field should be normalized according to the square root of `iter`, and speckle correlation is normalized according to `iter`.

<a name="ref"></a>
## References
Monte Carlo algorithm for far-field sources and sensors.

    @article{Bar2019MonteCarlo,
     author = {Bar, Chen and Alterman, Marina and Gkioulekas, Ioannis and Levin, Anat},
     title = {A Monte Carlo Framework for Rendering Speckle Statistics in Scattering Media},
     journal = {ACM Trans. Graph.},
     volume = {38},
     number = {4},
     year = {2019},
     month = {jul},
     doi = {10.1145/3306346.3322950}
    }

Monte Carlo algorithm for focused sources and sensors.

    @article{Bar2020Rendering,
     author = {Bar, Chen and Gkioulekas, Ioannis and Levin, Anat},
     title = {Rendering Near-Field Speckle Statistics in Scattering Media},
     journal = {ACM Trans. Graph.},
     volume = {39},
     number = {6},
     year = {2020},
     month = {nov},
     doi = {10.1145/3414685.3417813}
    }

Spatio-temporal Monte Carlo algorithm.

    @article{TBD,

    }

