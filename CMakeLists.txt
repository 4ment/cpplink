cmake_minimum_required(VERSION 3.15)
project(cpplink VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTING "Enable building of tests" OFF)
option(CPPLINK_BUILD_PYTHON "Build the Python extension module" OFF)

# Without this the default build has no -O flag at all, which costs roughly two
# orders of magnitude in the pair-evaluation loop. Debug builds must be asked for.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(nlohmann_json REQUIRED)

# Core logic lives in a library so both the CLI and the tests can link against it.
file(GLOB_RECURSE CPPLINK_SOURCES CONFIGURE_DEPENDS src/cpplink/*.cpp src/cpplink/*.hpp)

add_library(cpplink_core ${CPPLINK_SOURCES})
target_include_directories(cpplink_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# The Python extension is a shared module linking this static library, so its
# objects must be relocatable.
set_target_properties(cpplink_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cpplink_core PUBLIC
    Arrow::arrow_shared
    Parquet::parquet_shared
    nlohmann_json::nlohmann_json
)

add_executable(cpplink src/main.cpp)
target_link_libraries(cpplink PRIVATE cpplink_core)

if(CPPLINK_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(BUILD_TESTING)
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
