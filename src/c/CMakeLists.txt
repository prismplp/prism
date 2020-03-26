# if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
#   message(FATAL_ERROR "DO NOT BUILD in-tree.")
# endif()

cmake_minimum_required(VERSION 2.8)

project(prism_up_linux.bin)

# include(CheckPIESupported)
# check_pie_supported()

find_package(PkgConfig REQUIRED)
pkg_check_modules(HDF5 hdf5-serial REQUIRED)
link_directories(${HDF5_LIBRARY_DIRS})
include_directories(${HDF5_INCLUDE_DIRS})
pkg_check_modules(PROTOBUF protobuf REQUIRED)
link_directories(${PROTOBUF_LIBRARY_DIRS})
include_directories(${PROTOBUF_INCLUDE_DIRS})

add_executable(prism_up_linux.bin
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
target_compile_options(prism_up_linux.bin PUBLIC -fomit-frame-pointer -fno-strict-aliasing -Wall -DMALLOC_TRACE -DLINUX -DPOSIX -Dunix -DM64BITS -m64)
target_compile_features(prism_up_linux.bin PUBLIC cxx_std_11)

target_include_directories(prism_up_linux.bin
  PUBLIC
  ./bp4prism/include
  ./
  ./resource
  ./up
  )

target_link_libraries(prism_up_linux.bin PUBLIC m pthread hdf5_cpp ${HDF5_LIBRARIES} ${PROTOBUF_LIBRARIES} /prism/src/c/bp4prism/lib/bp4prism-linux.a -m64 -no-pie)

install(TARGETS prism_up_linux.bin RUNTIME DESTINATION /prism/bin)
