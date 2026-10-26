# SLM_hologram
**This is a mirror of the private `SLM_hologram` repo used in RbYbTweezerLab organization for public view with private info removed.**

Phase retrieval for Meadowlark SLM using (modified) Gerchberg-Saxton algorithm

<a name="top"></a>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Hardware and Environment](#hardware-and-environment)
- [Structure Overview](#structure-overview)
- [Configurations](#configurations)
- [Primer in Operation Theory](#primer-in-operation-theory)
  - [Fourier Optics and Phase Retrieval Problem](#fourier-optics-and-phase-retrieval-problem)
  - [Gerchberg-Saxton Algorithm](#gerchberg-saxton-algorithm)
  - [Modified Gerchberg-Saxton Algorithms](#modified-gerchberg-saxton-algorithms) 
  - [Reconstruct Fidelity](#reconstruct-fidelity)
- [Acceptable TCP Commands](#acceptable-tcp-commands)
- [Change Log and TODOs](#change-log-and-todos)
  - [TODOs](#todos)
- [More](#more)

## [Overview](#top)

This software architecture supports real-time phase retrieval using the Gerchberg-Saxton algorithm and its derivatives on a CUDA GPU platform, aiming to control 532 nm and 840 nm Meadowlark spatial-light modulators (SLMs) in the RbYb Tweezer Lab.

Currently, this project supports real-time hologram generation.
The following functions are expected to be added later:
1. Finalizing the automatic hologram optimization system with a feedback camera,
2. SLM-based tweezer rearrangement.

The project uses a basic TCP-based control logic that may be improved or replaced based on users' needs.  

## [Hardware and Environment](#top)

The program runs only with Meadowlark's SLM and Blink SDK on Windows and requires an NVIDIA GPU.

First, install the NVIDIA CUDA Toolkit and CUDA Driver. The code is tested with CUDA Toolkit 12.4. Make sure the cuFFT library is installed with the CUDA Toolkit.

The Blink SDK is provided directly by Meadowlark and should be installed by following their guide.
Note that the current SDK includes multiple changes; one major change is that the look-up table (LUT) files are now statistically written to the SLM using a separate program rather than the SDK.
Please refer to the `Blink_SDK_example.cpp` file within the SDK package for the newest SDK usage.

The image processing related code requires OpenCV.
Please follow the guide on their [official website](https://docs.opencv.org/4.5.1/df/d65/tutorial_table_of_content_introduction.html) for installation.

The project is developed as a Microsoft Visual Studio project in C++14 and CUDA C standard.
The tested program is compiled with MSVC19 and NVCC.

## [Structure Overview](#top)

- **holo_cuda.cu**  
  The main program controlling the SLM and computation flows

- **server.cpp**  
  The TCP server for receiving and deciphering commands from the host

- **image_process.cpp**  
  The detection and calibration code related with the external monitoring camera

- **gs_cuda.cu**  
  The CUDA kernels used by the GPU to execute the GS algorithm
  
- **CONFIGURATIONS.h**  
  The configuration file recording all options related to the SLM setting.
  One should set the parameters accordingly before compiling the program.
  See [Configurations](#configurations) for details.

## [Configurations](#top)

Since the SLM is expected to be part of a holographic system, the characteristics of the computer-generated hologram depend on the optical setup configuration.
As a result, one should modify the configurations written in `CONFIGURATIONS.h` accordingly before compiling this program to ensure hologram quality.
The current code assumes the incident beam is Gaussian and centered on the SLM surface.

`CONFIGURATIONS.h` contains both internal and configurable parameters, and the configurable ones are listed below. {wavelength} can be either 532 or 840:

- Optical Parameters
    - `PIXEL_WIDTH`/`PIXEL_HEIGHT`: The dimensions of the SLM surface in pixel counts
    - `PIXEL_PITCH`: The size of each SLM pixel in um
    - `BEAM_RADIUS_{wavelength}`: The $1/e^2$ radius of the incident Gaussian beam on the {wavelength} SLM in pixel counts
    - `WFC_FILE_PATH_{wavelength}`: The directory of the Wavefront Correction (WFC) file for {wavelength}
    - `FOCAL_LENGTH`: The focal length of the lens in um
    - `MAGNIFICATION_{wavelength}`: The magnification of the {wavelength} optical system before the lens
    - `WAVELENGTH_{wavelength}`: The wavelength of the beam in the {wavelength} optical system

- Software Parameters
    - `MAX_ITERATION`: The maximum iteration count for the GS algorithm
    - `IF_GS_THRESHOLD` (bool): Enable the threshold checker; allow the early termination of GS iterations with the cost of computation speed.
    - `GS_THRESHOLD`:  The threshold for satisfying GS convergence; need to be between 0 and 1; only effective when `IF_GS_THRESHOLD` enabled.
    - `TRIGGER_TIMEOUT`: The timeout for SLMs to receive a trigger in ms
    - `SLM_TYPE`: Which SLM is installed; can be either SLM_TYPE_532, SLM_TYPE_840, or SLM_TYPE_BOTH
    - `GS_MODE`: The mode of GS algorithm; can be either DIRECT, SELECTED, WEIGHTED, or SELECTED_WEIGHTED; see [Primer in Operation Theory](#primer-in-operation-theory) for details
    - `EXTERNAL_TRIGGER` (bool): Enable the external trigger for SLMs
    - `FLIP_IMMEDIATE` (bool): Enable forced update before previous phase pattern update for SLMs (not recommended)
    - `OUTPUT_PULSE` (bool): Enable a TTL output when phase pattern is updated for SLMs

- Dummy Test Parameters  
These parameters configure the dummy tweezer array pattern when there is no active command input
    - `OFFSET_X/Y_{wavelength}`: The position of the left/top corner of the dummy array for {wavelength} SLM in pixel counts
    - `DUMMY_PITCH_{wavelength}`: The array pitch for the dummy array for {wavelength} SLM in pixel counts
    - `DUMMY_SIDE_{wavelength}`: The number of tweezers per axis for the dummy array for {wavelength} SLM

- Image/Calibration Parameters
    - `SPOT_RADIUS`: The estimated spot $1/e^2$ radius of each tweezer spot on the monitoring camera in pixels
    - `BACKGROUND_INNER/OUTER_RADIUS`: The inner/outer radius of the annulus around each spot for local background sampling in pixels
    - `SPOT_MIN_DISTANCE`: The lower-bounded distance between spots in pixels; used to avoid double counting spots
    - `DISTINGUISH_THRESHOLD`: The intensity threshold for a sampled position being recognized as a spot; the value is in the unit of std/avg of intensity distributions, and the real threshold is the avg+`DISTINGUISH_THRESHOLD`*std
    - `IF_X/Y_INVERTED` (bool): If the x/y axis orientation in the image is inverted with respect to the programmed x/y position
    - `HOMOGENEITY_THRESHOLD`: the threshold for the std/avg of the tweezer intensities being considered "calibrated"
    - `CALIBRATION_FILENAME`:The name (directory) of the file where the position mapping from the programmed coordinates to the monitoring camera pixels is saved

## [Primer in Operation Theory](#top)

### [Fourier Optics and Phase Retrieval Problem](#primer-in-operation-theory)

The tweezer projection system can be simplified as a telescope system with a magnification $M$ before a converging objective of focal length $f$ that focuses the incident beams at the atoms' plane.
Ignoring the telescope and the aperture of the objective, the objective acts as a Fourier Transform operator between the transverse optical fields, $E(x,y)$ and $E'(x',y')$, at its two principal planes under paraxial approximations:

$$E(x,y)=\frac{e^{i\frac{4\pi}{\lambda}f}}{i\lambda f}\iint_\infty E'(x',y')e^{-i\frac{2\pi}{\lambda f}(xx'+yy')}dx'dy'=\frac{e^{i\frac{4\pi}{\lambda}f}}{i\lambda f}\mathcal{F}\left[E(x',y')\right] \left(\frac{x}{\lambda f},\frac{y}{\lambda f}\right)$$

where $\lambda$ is the wavelength of the light; $\mathcal{F}[T] (a,b)$ indicates the 2D Fourier Transform of $T$ with frequency $(a,b)$.

By placing a liquid crystal spatial light modulator (LC-SLM), which applies a programmable phase mask $\phi(x',y')$ to the reflected field $E(x',y')$, at the front principal plane, one can control the optical field at the back focal plane (atoms' plane).

Constrained by the fact that the programmable mask is phase-only, solving for the phase mask required to produce a specific intensity (amplitude) pattern at the focal plane does not have an exact, analytic solution.
This problem is termed the "phase retrieval" problem and is widely studied in engineering.

### [Gerchberg-Saxton Algorithm](#primer-in-operation-theory)
One well-adapted, numerical, iterative method to solve the problem is the Gerchberg-Saxton algorithm, invented by [R. W. Gerchberg and W. O. Saxton](https://antoine.wojdyla.fr/assets/archive/gerchberg_saxton1972.pdf) in 1971.
The method can be described as follows.  
To acquire a phase mask $\phi(x,y)$ for a fixed amplitude input $A(x,y)$ in the real space to generate a target amplitude pattern $A'(\omega_x,\omega_y)$ in Fourier space (the two patterns must have the same energy, i.e., $\iint_\infty A^2 dx\ dy=\iint_\infty A'^2d\omega_x\ d\omega_y$), one can start with a random phase mask $\phi_0(x,y)$ and iteratively repeat the following procedure:  
1. $B_n'(\omega_x,\omega_y)e^{i\phi'_n(\omega_x,\omega_y)}=\mathcal{F}\left[A(x,y)e^{i\phi_n(x,y)}\right] (\omega_x,\omega_y)$
2. $B_{n+1}(x,y)e^{i\phi_{n+1}(x,y)}=\mathcal{F}^{-1}\left[A'(\omega_x,\omega_y)e^{i\phi'_n(\omega_x,\omega_y)}\right] (x,y)$

where $B_n(x,y)$ and $B_n'(\omega_x,\omega_y)$ are intermediate amplitude patterns generated and replaced during the iterations.
As $n\rightarrow\infty$, $B_n(x,y)\rightarrow A(x,y)$, $B_n'(\omega_x,\omega_y)\rightarrow A_n'(\omega_x,\omega_y)$, and $\phi_n(x,y)$ will converges to a fixed pattern, which is the desired solution $\phi(x,y)$.

The GS algorithm is the fundamental algorithm used in this hologram program, and it is employed when the [configuration](#configurations) is set to `GS_MODE=GS_MODE_DIRECT`.

### [Modified Gerchberg-Saxton Algorithms](#primer-in-operation-theory)

The GS algorithm provides a general and simple solution to the phase retrieval problem, but it has several challenges.
For our purpose specifically, there are two major issues related to the algorithm.
First, when the GS algorithm is used to generate a highly symmetric/periodic spot array pattern, additional bright spots appear at undesired positions, referred to as "ghost spots/traps".
Second, for a pattern like a spot array, the GS algorithm needs about 10 iterations to stabilize fully; we would prefer faster convergence if we ever want to try tweezer rearrangement with an SLM.

As a result, I proposed two modified GS algorithms to handle the two problems individually.
To resolve the ghost trap problem, I referred to [Hesseling et al.'s](https://doi.org/10.1364/OL.36.003657) method, which I named the "selected GS algorithm".
When we substitute the intermediate amplitude patterns with the pre-determined amplitude pattern, we do so only on pixels whose amplitude is above a low threshold.
In this case, we allow a low-level background amplitude to evolve during the GS iteration and suppress it only where the intensity grows above the threshold.
This reduces the constraint placed on the phase pattern and thus resolves the ghost traps arising from intermodulations of the phase spectrum.

To accelerate the convergence of the GS, I, inspired by [Nogrette et al.'s](https://doi.org/10.1103/PhysRevX.4.021034) work, use a weighted GS algorithm, where instead of replacing $B_n(x,y)$ with $A(x,y)$, the substitute amplitude is calculated as following (same for $B_n'(\omega_x,\omega_y)$ and $A'(\omega_x,\omega_y)$ ):

$$A_n(x,y)=\frac{A(x,y)}{\sqrt{1-C\left(1-\frac{B_n^2(x,y)}{A_{n-1}^2(x,y)}\right)}}$$

I choose $C=0.8$ in the code to achieve fast convergence without oscillations.

It is worth noting that neither modification guarantees energy conservation, and I did not enforce conservation to maintain GPU computational efficiency.
In the realm of minor deviations from standard GS, the modifications do not really harm algorithm convergence.
If a problem prevents the algorithm from converging, I may add a normalizer to the code.

In the [configuration](#configurations) file, the selected GS algorithm is enabled when `GS_MODE=GS_MODE_SELECT`; the weighted GS algorithm is enabled when `GS_MODE=GS_MODE_WEIGHTED`; a combination of the two modifications is enabled when `GS_MODE=GS_MODE_SELECTED_WEIGHTED`.

### [Reconstruct Fidelity](#top)

To know if the generated hologram $B(x,y)$ is satisfactory enough to reconstruct target hologram $A(x,y)$, a fidelity checker is implemented to calculate the F1 score as follows:

$$F=\frac{2\sum |A(x,y)|^2|B(x,y)|^2}{\sum|A(x,y)|^4+\sum|B(x,y)|^4}$$

In the [configuration](#configurations) file, one can enable `IF_GS_THRESHOLD` and set a `GS_THRESHOLD` to allow the GS iteration halts when the fidelity is above the threshold, before reaching `MAX_ITERATION` number.

## [Acceptable TCP Commands](#top)

Currently, the program can only generate a static hologram.
The valid TCP string commands are:

- `{532/840},0,N,X0,X1...,Xn,Y0,Y1,...,Yn,A0,A1,...,An,go`:  
Generate a hologram with `N` single-pixel spots in the `532/840` path.
The position (in pixels) and relative amplitude of the spot are determined by `Xn,Yn,An`.

- `532,....,840,....,go`:  
Generate holograms on both paths simultaneously.
The grammar is similar to the single-path command above.

- `zero`:  
Generate a flat phase mask on both paths.

- `dummy`:  
Generate a pre-programmed, dummy hologram on both paths.

- `position_calibration`:  
Execute a calibration on the position mapping between the programmed coordinate and the monitoring camera's pixel coordinate.
This only works when only one of the SLMs is running with a generated hologram.
The calibration will only start after an image is acquired by the monitoring camera, either software or hardware triggered.
The generated calibration file will be saved locally at `CALIBRATION_FILENAME`.

- `intensity_calibration`:  
Execute a calibration on the intensity homogeneity of the spots for a generated hologram.
This only works when only one of the SLMs is running with a generated hologram.
The calibration will only start after an image is acquired by the monitoring camera, either software or hardware triggered.
Each call only executes one calibration iteration and returns the standard deviation of the average spot intensities.

- `{wavelength},{filename},save_phase`:  
Save the current phase mask on the `{wavelength}` path SLM to `{filename}`.

- `{wavelength},{filename},load_phase`:  
Load a phase mask from `{filename}` to the `{wavelength}` path SLM.

- `load_map`:  
Load a locally saved position map generated from previous position calibration.

- `stop`:  
Terminate the program.


## [Change Log and TODOs](#top)

### [TODOs](#change-log-and-todos)
1. Consider implementing a target hologram with Gaussian-like spots instead of single-pixel spots;
2. Attempt an SLM-based rearrangement protocol.


## [More](#top)

For RbYb Tweezer Lab members, please refer to the [OneNote page](https://onedrive.live.com/view.aspx?resid=601343494F74D454%215914&id=documents&wd=target%28Setting%20up%20Hardware%20Vol.%203.one%7CD3AEB2ED-07E7-4274-98D6-D8C3148A81F2%2FOliver%27s%20Note%7CC65A357D-C907-42AB-9E7B-8491C9C5001F%2F%29&wdpartid={97DD4065-6320-4E1F-AEB1-DCAD2A1F9F76}{1}&wdsectionfileid=601343494F74D454!8579&end) about the details of the program. Otherwise, please contact Juntian Tu (juntian"at"umd.edu) for issues related to this repository.
