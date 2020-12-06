# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lucalav/dace/dace/transformation/subgraph/stencil/manual

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lucalav/dace/dace/transformation/subgraph/stencil/manual/src

# Include any dependencies generated for this target.
include CMakeFiles/experimentallib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/experimentallib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/experimentallib.dir/flags.make

CMakeFiles/experimentallib.dir/kernels_experimental.cu.o: CMakeFiles/experimentallib.dir/flags.make
CMakeFiles/experimentallib.dir/kernels_experimental.cu.o: kernels_experimental.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lucalav/dace/dace/transformation/subgraph/stencil/manual/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/experimentallib.dir/kernels_experimental.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/lucalav/dace/dace/transformation/subgraph/stencil/manual/src/kernels_experimental.cu -o CMakeFiles/experimentallib.dir/kernels_experimental.cu.o

CMakeFiles/experimentallib.dir/kernels_experimental.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/experimentallib.dir/kernels_experimental.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/experimentallib.dir/kernels_experimental.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/experimentallib.dir/kernels_experimental.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target experimentallib
experimentallib_OBJECTS = \
"CMakeFiles/experimentallib.dir/kernels_experimental.cu.o"

# External object files for target experimentallib
experimentallib_EXTERNAL_OBJECTS =

CMakeFiles/experimentallib.dir/cmake_device_link.o: CMakeFiles/experimentallib.dir/kernels_experimental.cu.o
CMakeFiles/experimentallib.dir/cmake_device_link.o: CMakeFiles/experimentallib.dir/build.make
CMakeFiles/experimentallib.dir/cmake_device_link.o: CMakeFiles/experimentallib.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lucalav/dace/dace/transformation/subgraph/stencil/manual/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/experimentallib.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/experimentallib.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/experimentallib.dir/build: CMakeFiles/experimentallib.dir/cmake_device_link.o

.PHONY : CMakeFiles/experimentallib.dir/build

# Object files for target experimentallib
experimentallib_OBJECTS = \
"CMakeFiles/experimentallib.dir/kernels_experimental.cu.o"

# External object files for target experimentallib
experimentallib_EXTERNAL_OBJECTS =

libexperimentallib.so: CMakeFiles/experimentallib.dir/kernels_experimental.cu.o
libexperimentallib.so: CMakeFiles/experimentallib.dir/build.make
libexperimentallib.so: CMakeFiles/experimentallib.dir/cmake_device_link.o
libexperimentallib.so: CMakeFiles/experimentallib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lucalav/dace/dace/transformation/subgraph/stencil/manual/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA shared library libexperimentallib.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/experimentallib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/experimentallib.dir/build: libexperimentallib.so

.PHONY : CMakeFiles/experimentallib.dir/build

CMakeFiles/experimentallib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/experimentallib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/experimentallib.dir/clean

CMakeFiles/experimentallib.dir/depend:
	cd /home/lucalav/dace/dace/transformation/subgraph/stencil/manual/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lucalav/dace/dace/transformation/subgraph/stencil/manual /home/lucalav/dace/dace/transformation/subgraph/stencil/manual /home/lucalav/dace/dace/transformation/subgraph/stencil/manual/src /home/lucalav/dace/dace/transformation/subgraph/stencil/manual/src /home/lucalav/dace/dace/transformation/subgraph/stencil/manual/src/CMakeFiles/experimentallib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/experimentallib.dir/depend

