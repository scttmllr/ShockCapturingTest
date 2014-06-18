##
#  CMake script for AdvectionDG:
##

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

    #
    # Find and import the deal.II project configuration:
    #
FIND_PACKAGE(deal.II 8.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(AdvectionDG)

################################ User Variables ################################                              

#sets the dimension for the case
add_definitions( -D deal_II_dimension=2)

# set debug or release
set(CMAKE_BUILD_TYPE "debug")

#############################################################################

INCLUDE_DIRECTORIES(include/base
					  include/fluid)

SET(CMAKE_CXX_FLAGS ${DEAL_II_CXX_FLAGS})
SET(CMAKE_CXX_FLAGS_RELEASE ${DEAL_II_CXX_FLAGS_RELEASE})
SET(CMAKE_CXX_FLAGS_DEBUG ${DEAL_II_CXX_FLAGS_DEBUG})

FILE(GLOB MyCSources
source/*.cc)

ADD_LIBRARY(AdvectionDGSources SHARED ${MyCSources})

DEAL_II_SETUP_TARGET(AdvectionDGSources) 

ADD_EXECUTABLE(AdvectionDG source/main.cc)

DEAL_II_SETUP_TARGET(AdvectionDG) 

TARGET_LINK_LIBRARIES(AdvectionDG AdvectionDGSources)

SET_TARGET_PROPERTIES(AdvectionDG PROPERTIES
      LINK_FLAGS
      "${DEAL_II_LINKER_FLAGS}"
      LINK_FLAGS_DEBUG
      "${DEAL_II_LINKER_FLAGS_DEBUG}"
      LINK_FLAGS_RELEASE
      "${DEAL_II_LINKER_FLAGS_RELEASE}"
      )
