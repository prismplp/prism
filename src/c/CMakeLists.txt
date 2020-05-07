# if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
#   message(FATAL_ERROR "DO NOT BUILD in-tree.")
# endif()

cmake_minimum_required(VERSION 3.13)
project(prism_bin)

if (APPLE)
  set(CMAKE_CXX_FLAGS "-stdlib=libc++")
endif()
set (CMAKE_CXX_STANDARD 11)

if (APPLE)
  set(VAR_PLATFORM darwin)
elseif (CYGWIN)
  set(VAR_PLATFORM cygwin)
elseif (UNIX AND NOT APPLE AND NOT CYGWIN)
  set(VAR_PLATFORM linux)
endif()
message(${VAR_PLATFORM})

set(EXECUTABLE_1 prism_up_${VAR_PLATFORM}.bin)
if (CYGWIN)
  set(EXECUTABLE_1 prism_up_${VAR_PLATFORM}.exe)
endif()

# include(CheckPIESupported)
# check_pie_supported()

find_package(PkgConfig REQUIRED)
find_package(HDF5 REQUIRED)
# pkg_check_modules(HDF5 hdf5-serial REQUIRED)
link_directories(${HDF5_LIBRARY_DIRS})
include_directories(${HDF5_INCLUDE_DIRS})
pkg_check_modules(PROTOBUF protobuf REQUIRED)
link_directories(${PROTOBUF_LIBRARY_DIRS})
include_directories(${PROTOBUF_INCLUDE_DIRS})

add_executable(${EXECUTABLE_1}
  core/glue.c
  core/bpx.c
  core/idtable.c
  core/idtable_preds.c
  core/termpool.c
  core/vector.c
  core/random.c
  core/gamma.c
  core/xmalloc.c
  core/fputil.c
  core/error.c

  up/graph.c
  up/graph_aux.c
  up/em_preds.c
  up/em_ml.c
  up/em_vb.c
  up/em_aux.c
  up/em_aux_ml.c
  up/em_aux_vb.c
  up/vt_preds.c
  up/vt_ml.c
  up/vt_vb.c
  up/vt_aux_ml.c
  up/vt_aux_vb.c
  up/mcmc_preds.c
  up/mcmc_sample.c
  up/mcmc_eml.c
  up/mcmc_predict.c
  up/mcmc_exact.c
  up/viterbi.c
  up/hindsight.c
  up/flags.c
  up/crf_learn.c
  up/crf_viterbi.c
  up/crf_rank.c
  up/util.c
  up/linear_eq.cpp
  up/lbfgs.c
  up/scc.cpp
  up/nonlinear_eq.cpp
  up/cyc_em.cpp
  up/cyc_vt_preds.c
  up/cyc_vt_ml.cpp
  up/rank.cpp
  up/save_expl_graph.cpp
  up/tensor_preds.cpp
  up/sgd.cpp

  external/expl.pb.cc
)

# set_property(TARGET prism_up_linux.bin PROPERTY POSITION_INDEPENDENT_CODE FALSE)
target_compile_options(${EXECUTABLE_1} PUBLIC
  $<$<PLATFORM_ID:Darwin>:
    -fomit-frame-pointer -fno-strict-aliasing -Wall -arch x86_64 -Wno-c++11-narrowing -mmacosx-version-min=10.7
  >
  $<$<PLATFORM_ID:Linux>:
    -fomit-frame-pointer -fno-strict-aliasing -Wall -m64
  >
  $<$<PLATFORM_ID:CYGWIN>:
    -fomit-frame-pointer -fno-strict-aliasing -Wall
  >
)

# target_compile_features(prism_up_linux.bin PUBLIC cxx_std_11)
target_compile_definitions(${EXECUTABLE_1} PUBLIC
  $<$<PLATFORM_ID:Darwin>:
    DARWIN
  >
  $<$<PLATFORM_ID:Linux>:
    MALLOC_TRACE LINUX POSIX unix M64BITS
  >
  $<$<PLATFORM_ID:CYGWIN>:
    CYGWIN LINUX POSIX unix
  >
)

target_link_options(${EXECUTABLE_1} PUBLIC
  $<$<PLATFORM_ID:Darwin>:
    -arch x86_64 -mmacosx-version-min=10.7
  >
  $<$<PLATFORM_ID:Linux>:
    -m64 -no-pie
  >
)

target_include_directories(${EXECUTABLE_1}
  PUBLIC
  ./bp4prism/include
  ./
  ./resource
  ./up
  # /Library/Developer/CommandLineTools/usr/include/c++/v1
  )

set(BP4PRISM_A ${CMAKE_CURRENT_SOURCE_DIR}/bp4prism/lib/bp4prism-${VAR_PLATFORM}.a)
target_link_libraries(${EXECUTABLE_1} PUBLIC m pthread hdf5_cpp ${HDF5_LIBRARIES} ${PROTOBUF_LIBRARIES} ${BP4PRISM_A})

install(TARGETS ${EXECUTABLE_1} RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../../bin)
