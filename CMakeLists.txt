cmake_minimum_required(VERSION 3.15)
project(cpplink VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTING "Enable building of tests" OFF)
option(CPPLINK_BUILD_PYTHON "Build the Python extension module" OFF)
# Arrow C++ is what reads and writes parquet, and nothing else: every column
# crosses into the core through the C Data Interface, whose structs are vendored.
# Off, the parquet entry points fail with the reason and the library links no
# Arrow, which is how the Python wheel is built -- its input and output are data
# frames. AUTO takes Arrow where find_package finds it, which is the conda
# environment, and does without it where it does not, which is a wheel build.
# The tests need parquet fixtures, so they need it on.
set(CPPLINK_WITH_ARROW AUTO CACHE STRING "Read and write parquet through Arrow C++: ON, OFF or AUTO")
set_property(CACHE CPPLINK_WITH_ARROW PROPERTY STRINGS AUTO ON OFF)

# Without this the default build has no -O flag at all, which costs roughly two
# orders of magnitude in the pair-evaluation loop. Debug builds must be asked for.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Sanitizers, as a comma-separated -fsanitize list ("address,undefined", "thread").
# Applied to every target so the tests and the CLI run instrumented; the presets
# in CMakePresets.json set it. Undefined behaviour aborts rather than being
# printed and recovered from, so a test that trips it fails.
set(CPPLINK_SANITIZE "" CACHE STRING
    "Sanitizers to build with: a comma-separated -fsanitize= list, empty for none")
if(CPPLINK_SANITIZE)
    add_compile_options(-fsanitize=${CPPLINK_SANITIZE} -fno-omit-frame-pointer)
    add_link_options(-fsanitize=${CPPLINK_SANITIZE})
    if(CPPLINK_SANITIZE MATCHES "undefined")
        add_compile_options(-fno-sanitize-recover=undefined)
    endif()
endif()

# nlohmann_json is header-only: the environment's where it has one, else the
# pinned release, fetched, which is what a wheel build on a bare runner does.
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa)
    FetchContent_MakeAvailable(nlohmann_json)
    message(STATUS "nlohmann_json: fetched v3.12.0")
endif()

if(CPPLINK_WITH_ARROW STREQUAL "AUTO")
    find_package(Arrow QUIET)
    find_package(Parquet QUIET)
    if(Arrow_FOUND AND Parquet_FOUND)
        set(CPPLINK_WITH_ARROW ON)
    else()
        set(CPPLINK_WITH_ARROW OFF)
    endif()
    message(STATUS "Arrow: ${CPPLINK_WITH_ARROW} (found automatically)")
elseif(CPPLINK_WITH_ARROW)
    find_package(Arrow REQUIRED)
    find_package(Parquet REQUIRED)
endif()

# Core logic lives in a library so both the CLI and the tests can link against it.
file(GLOB_RECURSE CPPLINK_SOURCES CONFIGURE_DEPENDS src/cpplink/*.cpp src/cpplink/*.hpp)
# `parquet_io.hpp` has two implementations and a build compiles one of them.
if(CPPLINK_WITH_ARROW)
    list(FILTER CPPLINK_SOURCES EXCLUDE REGEX "parquet_io_stub\\.cpp$")
else()
    list(FILTER CPPLINK_SOURCES EXCLUDE REGEX "parquet_io_arrow\\.cpp$")
endif()

add_library(cpplink_core ${CPPLINK_SOURCES})
target_include_directories(cpplink_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# The Python extension is a shared module linking this static library, so its
# objects must be relocatable.
set_target_properties(cpplink_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cpplink_core PUBLIC nlohmann_json::nlohmann_json)
if(CPPLINK_WITH_ARROW)
    target_link_libraries(cpplink_core PUBLIC Arrow::arrow_shared Parquet::parquet_shared)
endif()

add_executable(cpplink src/main.cpp)
target_link_libraries(cpplink PRIVATE cpplink_core)

if(CPPLINK_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(BUILD_TESTING)
    if(NOT CPPLINK_WITH_ARROW)
        message(FATAL_ERROR "the tests write parquet fixtures: BUILD_TESTING needs CPPLINK_WITH_ARROW=ON")
    endif()
    enable_testing()

    find_package(GTest REQUIRED)
    include(GoogleTest)

    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
    add_executable(run_tests ${TEST_SOURCES})
    target_link_libraries(run_tests PRIVATE cpplink_core GTest::gtest_main)
    # A test reading a checked-in fixture names it from the source tree, so the
    # suite passes from ctest's working directory as well as from the root.
    target_compile_definitions(run_tests PRIVATE
        CPPLINK_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    # Arrow pulls in a large shared library graph; discovery needs longer than the
    # 5 s default to load it.
    gtest_discover_tests(run_tests DISCOVERY_TIMEOUT 120)
endif()

# clang-format target
find_program(CLANG_FORMAT_BIN clang-format)
file(GLOB PYTHON_BINDING_SOURCES CONFIGURE_DEPENDS python/bindings/*.cpp
     python/bindings/*.hpp)

if(CLANG_FORMAT_BIN)
    add_custom_target(
        format
        COMMAND ${CLANG_FORMAT_BIN} -i ${CPPLINK_SOURCES} src/main.cpp ${TEST_SOURCES}
                ${PYTHON_BINDING_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-format on source files"
    )
else()
    message(WARNING "clang-format not found")
endif()

install(TARGETS cpplink RUNTIME DESTINATION bin)
