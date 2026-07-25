set(OP_NAME nesterov_update)
set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

set(LINK_LIBS
  utility_cxx
  OpenMP::OpenMP_CXX
  )

if(TORCH_ENABLE_CUDA)
add_pytorch_extension(${TARGET_NAME}_cuda
  src/nesterov_update_cuda.cpp
  src/nesterov_update_cuda_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS
  ${TARGET_NAME}_cuda
  DESTINATION dreamplace/ops/${OP_NAME})
endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
