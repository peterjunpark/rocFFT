# rocFFT

rocFFT is a software library for computing fast Fourier transforms (FFTs) written in the HIP
programming language. It's part of AMD's software ecosystem based on
[ROCm](https://github.com/ROCm/ROCm). The rocFFT library can be used with AMD and
NVIDIA GPUs.

## Documentation

> [!NOTE]
> The published rocFFT documentation is available at [rocFFT](https://rocm.docs.amd.com/projects/rocFFT/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the rocFFT/docs folder of this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

To build our documentation locally, use the following code:

```Bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Build and install

You can install rocFFT using pre-built packages or building from source.

* Installing pre-built packages:

    1. Download the pre-built packages from the
        [ROCm package servers](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) or use the
        GitHub releases tab to download the source (this may give you a more recent version than the
        pre-built packages).

    2. Run: `sudo apt update && sudo apt install rocfft`

* Building from source:

    rocFFT is compiled with AMD's clang++ and uses CMake. You can specify several options to customize your
    build. The following commands build a shared library for supported AMD GPUs:

    ```bash
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang ..
    make -j
    ```

    You can compile a static library using the `-DBUILD_SHARED_LIBS=off` option.

    With rocFFT, you can use indirect function calls by default; this requires ROCm 4.3 or higher. You can
    use `-DROCFFT_CALLBACKS_ENABLED=off` with CMake to prevent these calls on older ROCm
    compilers. Note that with this configuration, callbacks won't work correctly.

    rocFFT includes the following clients:

  * `rocfft-bench`: Runs general transforms and is useful for performance analysis
  * `rocfft-test`: Runs various regression tests
  * Various small samples

    | Client | CMake option | Dependencies |
    |:------|:-----------------|:-----------------|
    | `rocfft-bench` | `-DBUILD_CLIENTS_BENCH=on` | None |
    | `rocfft-test` | `-DBUILD_CLIENTS_TESTS=on` | Fastest Fourier Transform in the West (FFTW), GoogleTest |
    | samples | `-DBUILD_CLIENTS_SAMPLES=on` | FFTW |

    Clients are not built by default. To build them, use `-DBUILD_CLIENTS=on`. The build process
    downloads and builds GoogleTest and FFTW if they are not already installed.

    Clients can be built separately from the main library. For example, you can build all the clients with
    an existing rocFFT library by invoking CMake from within the `rocFFT-src/clients` folder:

    ```bash
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang_PREFIX_PATH=/path/to/rocFFT-lib ..
    make -j
    ```

    To install client dependencies on Ubuntu, run:

    ```bash
    sudo apt install libgtest-dev libfftw3-dev
    ```

    We use version 1.11 of GoogleTest.

## Examples

A summary of the latest functionality and workflow to compute an FFT with rocFFT is available on the
[rocFFT documentation portal](https://rocm.docs.amd.com/projects/rocFFT/en/latest/).

You can find additional examples in the `clients/samples` subdirectory.

## Support

You can report bugs and feature requests through the GitHub
[issue tracker](https://github.com/ROCm/rocFFT/issues).

## Contribute

If you want to contribute to rocFFT, you must follow our [contribution guidelines](https://github.com/ROCm/rocFFT/blob/develop/.github/CONTRIBUTING.md).
