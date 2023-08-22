cmake_minimum_required(VERSION 3.14)
project(triangulation-generator VERSION 0.1 LANGUAGES CXX)

if(APPLE)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()


# Define options
option(BUILD_PY_BINDINGS    "Builds Python bindings"        ON)
option(DEBUG_MODE           "Adds debug flag to compiler"   OFF)
option(SANITIZE_ADDRESS     "Enables address sanitizer"     OFF)

# Set to C++14
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++14")
set(CMAKE_BUILD_TYPE DEBUG)

if (DEBUG_MODE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()  # DEBUG MODE

if (SANITIZE_ADDRESS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()  # SANITIZE_ADDRESS


############# LOAD PACKAGES #############

# CGAL and its components
find_package(CGAL QUIET COMPONENTS)

if (NOT CGAL_FOUND)
  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()
endif()  # CGAL_FOUND


# Boost and its components
# set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

if (BUILD_PY_BINDINGS)
    find_package(Boost COMPONENTS python REQUIRED)
    find_package (Python COMPONENTS Interpreter Development)
endif()  # BUILD_PY_BINDINGS

if (NOT Boost_FOUND)
  message(STATUS "This project requires the Boost library, and will not be compiled.")
  return()
endif()  # Boost_FOUND

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    /opt/homebrew/include/eigen3)


## Add executables
add_executable(weighted weighted.cpp)
target_link_libraries(weighted CGAL::CGAL ${Boost_PROGRAM_OPTIONS_LIBRARY})

if (BUILD_PY_BINDINGS)
    add_library(py_polytope MODULE py_polytope.cpp)
    target_link_libraries(py_polytope CGAL::CGAL Boost::python ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_LIBRARIES} Python::Python)
endif()  # BUILD_PY_BINDINGS
