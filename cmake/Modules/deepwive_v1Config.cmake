INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_DEEPWIVE_V1 deepwive_v1)

FIND_PATH(
    DEEPWIVE_V1_INCLUDE_DIRS
    NAMES deepwive_v1/api.h
    HINTS $ENV{DEEPWIVE_V1_DIR}/include
        ${PC_DEEPWIVE_V1_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    DEEPWIVE_V1_LIBRARIES
    NAMES gnuradio-deepwive_v1
    HINTS $ENV{DEEPWIVE_V1_DIR}/lib
        ${PC_DEEPWIVE_V1_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/deepwive_v1Target.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DEEPWIVE_V1 DEFAULT_MSG DEEPWIVE_V1_LIBRARIES DEEPWIVE_V1_INCLUDE_DIRS)
MARK_AS_ADVANCED(DEEPWIVE_V1_LIBRARIES DEEPWIVE_V1_INCLUDE_DIRS)
